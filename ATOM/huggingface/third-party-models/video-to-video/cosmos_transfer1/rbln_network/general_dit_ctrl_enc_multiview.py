# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import rebel
import torch
from cosmos_transfer1.diffusion.conditioner import DataType
from cosmos_transfer1.utils import log
from einops import rearrange
from torch import nn
from torchvision import transforms

from .general_dit_ctrl_enc import RBLNRuntimeControlNet
from .utils import run_model


class RBLNGeneralDITMultiviewEncoder:
    def __init__(
        self,
        hint_encoders,
        control_input_keys,
        in_channels,
        use_cross_attn_mask,
    ):
        self.hint_encoders = hint_encoders
        self.control_input_keys = control_input_keys
        self.in_channels = in_channels
        self.use_cross_attn_mask = use_cross_attn_mask
        self.is_context_parallel_enabled = False
        self.cp_group = None

    def create_runtime(self):
        for hint_encoder in self.hint_encoders.values():
            hint_encoder.create_runtime()

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        hint_key: Optional[str] = None,
        base_model: Optional[nn.Module] = None,
        control_weight: Optional[float] = 1.0,
        num_layers_to_use: Optional[int] = -1,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
        """
        # record the input as they are replaced in this forward
        x_input = x
        frame_repeat = kwargs.pop("frame_repeat", None)
        crossattn_emb_input = crossattn_emb
        crossattn_mask_input = crossattn_mask
        condition_video_input_mask_input = condition_video_input_mask
        hint = kwargs.pop(hint_key)
        if "multi" not in hint_key:
            hint = hint.unsqueeze(1)
        if hint is None:
            log.info("using none hint")
            return base_model.net.forward(
                x=x_input,
                timesteps=timesteps,
                crossattn_emb=crossattn_emb_input,
                crossattn_mask=crossattn_mask_input,
                fps=fps,
                image_size=image_size,
                padding_mask=padding_mask,
                scalar_feature=scalar_feature,
                data_type=data_type,
                condition_video_input_mask=condition_video_input_mask_input,
                latent_condition=latent_condition,
                latent_condition_sigma=latent_condition_sigma,
                view_indices_B_T=view_indices_B_T,
                **kwargs,
            )

        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )

        B, C, T, H, W = x.shape
        if data_type == DataType.VIDEO:
            if condition_video_input_mask is not None:
                input_list = [x, condition_video_input_mask]
                x = torch.cat(
                    input_list,
                    dim=1,
                )

        elif data_type == DataType.IMAGE:
            # For image, we dont have condition_video_input_mask, or condition_video_pose
            # We need to add the extra channel for video condition mask
            padding_channels = self.in_channels - x.shape[1]
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        (B, padding_channels, T, H, W), dtype=x.dtype, device=x.device
                    ),
                ],
                dim=1,
            )
        else:
            assert x.shape[1] == self.in_channels, (
                f"Expected {self.in_channels} channels, got {x.shape[1]}"
            )

        self.crossattn_emb = crossattn_emb
        self.crossattn_mask = crossattn_mask

        if self.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(
                dtype=torch.bool
            )  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        crossattn_emb = rearrange(crossattn_emb, "B M D -> M B D")
        if crossattn_mask:
            crossattn_mask = rearrange(crossattn_mask, "B M -> M B")

        x_ctrl = None

        if isinstance(control_weight, torch.Tensor):
            if control_weight.ndim == 0:  # Single scalar tensor
                control_weight = [float(control_weight)] * hint.shape[1]
            elif control_weight.ndim == 1:  # List of scalar weights
                control_weight = [float(w) for w in control_weight]
            else:  # Spatial-temporal weight maps
                control_weight = [w for w in control_weight]  # Keep as tensor
        else:
            control_weight = [control_weight] * hint.shape[1]

        x_before_blocks = x.clone()
        jobs = {}
        with ThreadPoolExecutor(max_workers=len(self.control_input_keys)) as executor:
            for i, k in enumerate(self.control_input_keys):
                hint_encoder = self.hint_encoders[k]
                x = x_before_blocks
                jobs[i] = executor.submit(
                    run_model,
                    hint_encoder,
                    x=x,
                    timesteps=timesteps,
                    hint=hint[:, i],
                    crossattn_emb=crossattn_emb,
                    crossattn_mask=crossattn_mask,
                    fps=fps,
                    padding_mask=padding_mask,
                    scalar_feature=scalar_feature,
                    data_type=data_type,
                    control_weight=control_weight[i],
                    num_layers_to_use=num_layers_to_use,
                    condition_video_input_mask=condition_video_input_mask,
                    latent_condition=latent_condition,
                    latent_condition_sigma=latent_condition_sigma,
                    view_indices_B_T=view_indices_B_T,
                    frame_repeat=frame_repeat,
                    **kwargs,
                )
            for i, k in enumerate(self.control_input_keys):
                outs = jobs[i].result()
                if x_ctrl is None:
                    x_ctrl = outs
                else:
                    x_ctrl = [x_ctrl[i] + outs[i] for i in range(len(outs))]

        output = base_model.net.forward(
            x=x_input,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb_input,
            crossattn_mask=crossattn_mask_input,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            x_ctrl=outs,
            condition_video_input_mask=condition_video_input_mask_input,
            latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
            view_indices_B_T=view_indices_B_T,
            **kwargs,
        )
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class RBLNRuntimeControlNetMultiview(RBLNRuntimeControlNet):
    def __init__(
        self,
        compiled_model: rebel.RBLNCompiledModel,
        net: nn.Module,
        rbln_config: Optional[Dict] = None,
        num_regions: Optional[int] = None,
    ):
        super().__init__(compiled_model, net, rbln_config, num_regions)

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        trajectory: Optional[torch.Tensor] = None,
        frame_repeat: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and
        handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for
                positional embedding when required. If None, a default value
                (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the
                    positional embedding class (`self.pos_emb_cls`) includes 'rope'.
                    Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is
                concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of
                `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional
                embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are
                generated using the `self.pos_embedder` with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering
                fps.
        """
        if self.net.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask,
                list(x_B_C_T_H_W.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            x_B_C_T_H_W = torch.cat(
                [
                    x_B_C_T_H_W,
                    padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1),
                ],
                dim=1,
            )

        if view_indices_B_T is None:
            view_indices = torch.arange(self.net.n_views).clamp(
                max=self.net.n_views_emb - 1
            )  # View indices [0, 1, ..., V-1]
            view_indices = view_indices.to(x_B_C_T_H_W.device)
            view_embedding = self.net.view_embeddings(
                view_indices
            )  # Shape: [V, embedding_dim]
            view_embedding = rearrange(view_embedding, "V D -> D V")
            view_embedding = (
                view_embedding.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)
            )  # Shape: [1, D, V, 1, 1, 1]
        else:
            view_indices_B_T = view_indices_B_T.clamp(max=self.net.n_views_emb - 1)
            view_indices_B_T = view_indices_B_T.to(x_B_C_T_H_W.device).long()
            view_embedding = self.net.view_embeddings(view_indices_B_T)  # B, (V T), D
            view_embedding = rearrange(
                view_embedding, "B (V T) D -> B D V T", V=self.net.n_views
            )
            view_embedding = view_embedding.unsqueeze(-1).unsqueeze(
                -1
            )  # Shape: [B, D, V, T, 1, 1]

        if self.net.add_repeat_frame_embedding:
            if frame_repeat is None:
                frame_repeat = (
                    torch.zeros([x_B_C_T_H_W.shape[0], view_embedding.shape[1]])
                    .to(view_embedding.device)
                    .to(view_embedding.dtype)
                )
            frame_repeat_embedding = self.net.repeat_frame_embedding(
                frame_repeat.unsqueeze(-1)
            )
            frame_repeat_embedding = rearrange(frame_repeat_embedding, "B V D -> B D V")
            view_embedding = view_embedding + frame_repeat_embedding.unsqueeze(
                3
            ).unsqueeze(4).unsqueeze(5)

        x_B_C_V_T_H_W = rearrange(
            x_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=self.net.n_views
        )
        view_embedding = view_embedding.expand(
            x_B_C_V_T_H_W.shape[0],
            view_embedding.shape[1],
            view_embedding.shape[2],
            x_B_C_V_T_H_W.shape[3],
            x_B_C_V_T_H_W.shape[4],
            x_B_C_V_T_H_W.shape[5],
        )  # Shape: [B, V, 3, t, H, W]
        if self.net.concat_traj_embedding:
            traj_emb = self.net.traj_embeddings(trajectory)
            traj_emb = traj_emb.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)
            traj_emb = traj_emb.expand(
                x_B_C_V_T_H_W.shape[0],
                traj_emb.shape[1],
                view_embedding.shape[2],
                x_B_C_V_T_H_W.shape[3],
                x_B_C_V_T_H_W.shape[4],
                x_B_C_V_T_H_W.shape[5],
            )  # Shape: [B, V, 3, t, H, W]

            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding, traj_emb], dim=1)
        else:
            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)

        x_B_C_T_H_W = rearrange(
            x_B_C_V_T_H_W, " B C V T H W -> B C (V T) H W", V=self.net.n_views
        )
        x_B_T_H_W_D = self.net.x_embedder(x_B_C_T_H_W)

        if self.net.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.net.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.net.pos_emb_cls.lower():
            rope_emb_L_1_1_D = self.net.pos_embedder(x_B_T_H_W_D, fps=fps)
            return x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb

        if "fps_aware" in self.net.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.net.pos_embedder(
                x_B_T_H_W_D, fps=fps
            )  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.net.pos_embedder(
                x_B_T_H_W_D
            )  # [B, T, H, W, D]
        return x_B_T_H_W_D, None, extra_pos_emb

    def prepare_hint_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.net.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask,
                list(x_B_C_T_H_W.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            x_B_C_T_H_W = torch.cat(
                [
                    x_B_C_T_H_W,
                    padding_mask.unsqueeze(1).repeat(
                        x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[2], 1, 1
                    ),
                ],
                dim=1,
            )

        x_B_T_H_W_D = self.net.x_embedder2(x_B_C_T_H_W)

        if "rope" in self.net.pos_emb_cls.lower():
            rope_emb_L_1_1_D = self.net.pos_embedder(x_B_T_H_W_D, fps=fps)
            return x_B_T_H_W_D, rope_emb_L_1_1_D

        if "fps_aware" in self.net.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.net.pos_embedder(
                x_B_T_H_W_D, fps=fps
            )  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.net.pos_embedder(
                x_B_T_H_W_D
            )  # [B, T, H, W, D]
        return x_B_T_H_W_D, None

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        hint: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        control_weight: Optional[float] = 1.0,
        num_layers_to_use: Optional[int] = -1,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
        frame_repeat: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        guided_hints = self.encode_hint(
            hint, fps=fps, padding_mask=padding_mask, data_type=data_type
        )
        guided_hints = torch.chunk(guided_hints, hint.shape[0] // x.shape[0], dim=3)
        # Only support multi-control at inference time
        assert len(guided_hints) == 1 or not torch.is_grad_enabled()

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = (
            self.prepare_embedded_sequence(
                x,
                fps=fps,
                padding_mask=padding_mask,
                latent_condition=latent_condition,
                latent_condition_sigma=latent_condition_sigma,
                frame_repeat=frame_repeat,
                view_indices_B_T=view_indices_B_T,
            )
        )
        # logging affline scale information
        affline_scale_log_info = {}

        timesteps_B_D, adaln_lora_B_3D = self.net.t_embedder(timesteps.flatten())
        affline_emb_B_D = timesteps_B_D
        affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

        if scalar_feature is not None:
            raise NotImplementedError("Scalar feature is not implemented yet.")

        affline_scale_log_info["affline_emb_B_D"] = affline_emb_B_D.detach()
        affline_emb_B_D = self.net.affline_norm(affline_emb_B_D)

        # for logging purpose
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = affline_emb_B_D

        x = rearrange(x_B_T_H_W_D, "B T H W D -> T H W B D")
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "B T H W D -> T H W B D"
            )

        output = self.runtime(
            guided_hints[0],
            x,
            rope_emb_L_1_1_D[0],
            rope_emb_L_1_1_D[1],
            extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            crossattn_emb,
            adaln_lora_B_3D,
            affline_emb_B_D,
            control_weight=torch.tensor([control_weight]),
        )
        return output
