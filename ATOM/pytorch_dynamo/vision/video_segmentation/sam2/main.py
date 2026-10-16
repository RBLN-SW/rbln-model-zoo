import argparse
import os
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(sys.path[0], "sam2"))

import rebel  # noqa: F401 # needed to use torch.compile with backend="rebel"
from sam2.modeling.sam.prompt_encoder import PromptEncoder as BasePromptEncoder
from sam2.modeling.sam.transformer import RoPEAttention as BaseRoPEAttention
from sam2.sam2_video_predictor import SAM2VideoPredictorVOS


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="sam2-hiera-large",
        help="(str) model name, e.g., sam2-hiera-large",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="(bool) whether to export the model",
    )
    return parser.parse_args()


class PromptEncoder(BasePromptEncoder):
    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        points = points + 0.5
        if pad:
            padding_point = torch.zeros(
                (points.shape[0], 1, 2), device=points.device, dtype=points.dtype
            )
            padding_label = -torch.ones(
                (labels.shape[0], 1), device=labels.device, dtype=labels.dtype
            )
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )

        for i, w in enumerate(
            [self.not_a_point_embed.weight]
            + [pe.weight for pe in self.point_embeddings]
        ):
            point_embedding = torch.where(
                (labels == (i - 1 if i else -1)).unsqueeze(-1),
                point_embedding + w,
                point_embedding,
            )
        return point_embedding

    def _get_batch_size(
        self,
        points_coords: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        return next(
            filter(lambda x: x is not None, [points_coords, boxes, masks]), 1
        ).shape[0]

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        points_coords, points_labels = points
        bs = self._get_batch_size(points_coords, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points_coords is not None:
            point_embeddings = self._embed_points(
                points_coords, points_labels, pad=(boxes is None)
            )
            sparse_embeddings = torch.cat(
                [sparse_embeddings.to(point_embeddings.dtype), point_embeddings], dim=1
            )
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


# in position encoding
def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    from sam2.modeling.position_encoding import init_t_xy

    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)

    freqs_cis_real = torch.cat([torch.cos(freqs_x), torch.cos(freqs_y)], dim=-1)
    freqs_cis_imag = torch.cat([torch.sin(freqs_x), torch.sin(freqs_y)], dim=-1)

    return freqs_cis_real, freqs_cis_imag


# in position encoding
def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    from sam2.modeling.position_encoding import reshape_for_broadcast
    from torch import Tensor

    def apply_rotary(x: Tensor, freqs_real: Tensor, freqs_imag: Tensor) -> Tensor:
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_real, x_imag = torch.split(x_reshaped, x_reshaped.shape[-1] // 2, dim=-1)
        x_real = x_real.squeeze(-1)
        x_imag = x_imag.squeeze(-1)
        freqs_real = freqs_real.squeeze().to(x.device)
        freqs_imag = freqs_imag.squeeze().to(x.device)
        freqs_real = reshape_for_broadcast(freqs_real, x_real)
        freqs_imag = reshape_for_broadcast(freqs_imag, x_imag)
        out_real = x_real * freqs_real - x_imag * freqs_imag
        out_imag = x_real * freqs_imag + x_imag * freqs_real
        return torch.stack([out_real, out_imag], dim=-1).flatten(start_dim=-2)

    freqs_real, freqs_imag = freqs_cis
    xq_out = apply_rotary(xq, freqs_real, freqs_imag)

    if xk.shape[-2] == 0:
        return xq_out.type_as(xq), xk

    if repeat_freqs_k and xq.shape[-2] > 0:
        r = xk.shape[-2] // xq.shape[-2]
        if r > 1:
            freqs_real = freqs_real.repeat(1, 1, r, 1)
            freqs_imag = freqs_imag.repeat(1, 1, r, 1)

    xk_out = apply_rotary(xk, freqs_real, freqs_imag)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPEAttention(BaseRoPEAttention):
    from torch import Tensor

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(64, 64),
        **kwargs,
    ):
        from functools import partial

        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        self.freqs_cis_real, self.freqs_cis_imag = self.compute_cis(
            end_x=feat_sizes[0], end_y=feat_sizes[1]
        )
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        import math

        import torch.nn.functional as F

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        w = h = math.sqrt(q.shape[-2])
        if (
            self.freqs_cis_real.shape[0] != q.shape[-2]
            or self.freqs_cis_imag.shape[0] != q.shape[-2]
        ):
            self.freqs_cis_real, self.freqs_cis_imag = self.compute_cis(
                end_x=w, end_y=h
            )

        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=(self.freqs_cis_real, self.freqs_cis_imag),
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


def build_sam2_video_predictor_hf(model_id, **kwargs):
    from sam2.build_sam import _hf_download

    config_name, ckpt_path = _hf_download(model_id)

    def build_sam2_video_predictor(
        config_file,
        ckpt_path=None,
        device="cuda",
        mode="eval",
        hydra_overrides_extra=[],
        apply_postprocessing=True,
        vos_optimized=False,
        **kwargs,
    ):
        from hydra import compose
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from sam2.build_sam import _load_checkpoint

        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
        ]
        if vos_optimized:
            relative_path = os.path.relpath(__file__, os.getcwd())
            file_name = relative_path.split(".")[0].replace("/", ".")
            file_name = os.path.basename(__file__).split(".")[0]

            hydra_overrides = [
                f"++model._target_={file_name}.RBLNSAM2VideoPredictor",
                "++model.compile_image_encoder=False",
            ]
            hydra_overrides += [
                f"++model.memory_attention.layer.self_attention._target_={file_name}.RoPEAttention",
                f"++model.memory_attention.layer.cross_attention._target_={file_name}.RoPEAttention",
            ]

        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                "++model.binarize_mask_from_pts_for_mem_enc=true",
                "++model.fill_hole_area=8",
            ]
        hydra_overrides.extend(hydra_overrides_extra)

        cfg = compose(config_name=config_file, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(
            cfg.model,
            options=kwargs.get("options", {"cache_dir": ".cache", "device": 0}),
            _recursive_=True,
        )
        _load_checkpoint(model, ckpt_path)
        model = model.to(device)
        if mode == "eval":
            model.eval()
        return model

    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


class RBLNSAM2VideoPredictor(SAM2VideoPredictorVOS):
    def __init__(
        self, *args, options: dict = {"cache_dir": ".cache", "device": 0}, **kwargs
    ):
        super().__init__(*args, **kwargs)
        torch._dynamo.config.recompile_limit = 16
        self._custom_compile_all_components(options=options)

    # Python doesn't support method overloading; calls are resolved by name, not arguments.
    def _compile_all_components(self):
        pass

    def _custom_compile_all_components(
        self, options: dict = {"cache_dir": ".cache", "device": 0}
    ):
        cache_dir = options["cache_dir"]
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.image_encoder.forward = torch.compile(
            self.image_encoder.forward,
            backend="rbln",
            options=options,
            dynamic=False,
        )
        self.memory_encoder.forward = torch.compile(
            self.memory_encoder.forward,
            backend="rbln",
            options=options,
            dynamic=False,
        )
        self.memory_attention.forward = torch.compile(
            self.memory_attention.forward,
            backend="rbln",
            options=options,
            dynamic=False,
        )
        self.sam_prompt_encoder.forward = torch.compile(
            self.sam_prompt_encoder.forward,
            backend="rbln",
            options=options,
            dynamic=False,
        )
        self.sam_mask_decoder.forward = torch.compile(
            self.sam_mask_decoder.forward,
            backend="rbln",
            options=options,
            dynamic=False,
        )

    def _build_sam_heads(self):
        super()._build_sam_heads()

        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        options: dict = {"cache_dir": ".cache", "device": 0},
        **kwargs,
    ):
        kwargs["device"] = "cpu"
        kwargs["vos_optimized"] = True
        kwargs["options"] = options
        sam_model = build_sam2_video_predictor_hf(model_id, **kwargs)

        return sam_model

    def to_export(self):
        import requests

        raw_video_url = "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/videos/bedroom.mp4"
        local_video_path = "tmp/bedroom.mp4"

        if not os.path.exists(os.path.dirname(local_video_path)):
            os.makedirs(os.path.dirname(local_video_path))

        response = requests.get(raw_video_url)
        if response.status_code == 200:
            with open(local_video_path, "wb") as f:
                f.write(response.content)
            print(f"Video downloaded to {local_video_path}")
        else:
            print("Failed to download video")
        inference_state = self.init_state(video_path=local_video_path)

        _, _, _ = self.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=np.array([[0, 0, 0, 0]], dtype=np.float32),
        )
        for out_frame_idx, _, _ in self.propagate_in_video(inference_state):
            if out_frame_idx > self.num_maskmem * 2 + 1:
                return


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def visualize_segments(
    video_segments, frame_names, video_dir, output_dir, vis_frame_stride=30
):
    os.makedirs(output_dir, exist_ok=True)

    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        plt.savefig(
            os.path.join(output_dir, f"segmented_object_{out_frame_idx}.jpg"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()


def main():
    args = parsing_argument()

    # Load model checkpoint
    checkpoint = os.path.join("facebook/", args.model_name)
    predictor = RBLNSAM2VideoPredictor.from_pretrained(
        checkpoint,
    )

    # Add support for explicit model compilation
    if args.export:
        return predictor.to_export()

    # Load video directory
    video_dir = "sam2/notebooks/videos/bedroom"
    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Initialize inference state
    inference_state = predictor.init_state(video_path=video_dir)

    ann_frame_idx = 0
    ann_obj_id = 1

    # Add new points
    points = np.array([[210, 350], [250, 220]], dtype=np.float32)
    labels = np.array([1, 1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # Propagate segmentation masks through the video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Visualize segmentation results
    visualize_segments(
        video_segments, frame_names, video_dir, output_dir="output_video_points"
    )

    # Reset inference state
    predictor.reset_state(inference_state)

    ann_frame_idx = 0
    ann_obj_id = 1

    # Add new box
    box = np.array([300, 0, 500, 400], dtype=np.float32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
    )

    # Propagate segmentation masks through the video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Visualize segmentation results
    visualize_segments(
        video_segments, frame_names, video_dir, output_dir="output_video_box"
    )


if __name__ == "__main__":
    main()
