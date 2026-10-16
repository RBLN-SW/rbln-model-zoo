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
import torch.nn as nn
from cosmos_transfer1.diffusion.conditioner import DataType
from einops import rearrange
from torchvision import transforms

from .utils import run_model


class RBLNGeneralDITEncoder:
    def __init__(
        self, hint_encoders, control_input_keys, in_channels, use_cross_attn_mask
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
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        hint_key: Optional[str] = None,
        base_model: Optional[nn.Module] = None,
        control_weight: Optional[float] = 1.0,
        num_layers_to_use: Optional[int] = -1,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        regional_contexts: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
        base_ratio: float = 0.5,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        x_input = x
        crossattn_emb_input = crossattn_emb
        crossattn_mask_input = crossattn_mask
        condition_video_input_mask_input = condition_video_input_mask

        regional_contexts_input = regional_contexts
        region_masks_input = region_masks

        hint = kwargs.pop(hint_key)
        if "multi" not in hint_key:
            hint = hint.unsqueeze(1)
        if hint is None:
            print("using none hint")
            return base_model.net.forward(
                x=x_input,
                timesteps=timesteps,
                crossattn_emb=crossattn_emb_input,
                crossattn_mask=crossattn_mask_input,
                fps=fps,
                padding_mask=padding_mask,
                scalar_feature=scalar_feature,
                data_type=data_type,
                condition_video_input_mask=condition_video_input_mask_input,
                regional_contexts=regional_contexts_input,
                region_masks=region_masks_input,
                base_ratio=base_ratio,
                **kwargs,
            )

        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )

        B, C, T, H, W = x.shape
        if data_type == DataType.VIDEO:
            if condition_video_input_mask is not None:
                input_list = [x, condition_video_input_mask]
                x = torch.cat(input_list, dim=1)
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
        if regional_contexts is not None:
            regional_contexts = rearrange(regional_contexts, "B R M D -> R M B D")
        if region_masks is not None:
            region_masks = rearrange(region_masks, "B R T H W -> R T H W B")

        x_ctrl = None

        if isinstance(control_weight, torch.Tensor):
            if control_weight.ndim == 0:  # Single scalar tensor
                control_weight = [float(control_weight)]
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
                    regional_contexts=regional_contexts,
                    region_masks=region_masks,
                    base_ratio=base_ratio,
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
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            x_ctrl=x_ctrl,
            condition_video_input_mask=condition_video_input_mask_input,
            regional_contexts=regional_contexts_input,
            region_masks=region_masks_input,
            base_ratio=base_ratio,
            **kwargs,
        )
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class RBLNRuntimeControlNet:
    def __init__(
        self,
        compiled_model: rebel.RBLNCompiledModel,
        net: nn.Module,
        rbln_config: Optional[Dict] = None,
        num_regions: Optional[int] = None,
    ):
        self.compiled_model = compiled_model
        self.net = net
        self.rbln_config = rbln_config
        self.num_regions = num_regions
        self.dtype = torch.float32
        self._runtime = None

    @property
    def runtime(self):
        if self._runtime is None:
            raise ValueError(
                "Runtime is not created. Please set `create_runtimes=True` first."
            )
        return self._runtime

    def create_runtime(self):
        if self._runtime is None:
            if self.rbln_config is None:
                device = None
            else:
                device = self.rbln_config.get("device", None)
            self._runtime = self.compiled_model.create_runtime(
                tensor_type="pt", device=device
            )
        else:
            print("Runtime is created already.")

    def encode_hint(
        self,
        hint: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
    ) -> torch.Tensor:
        assert hint.size(1) <= self.net.hint_channels, (
            f"Expected hint channels <= {self.net.hint_channels}, got {hint.size(1)}"
        )
        if hint.size(1) < self.net.hint_channels:
            padding_shape = list(hint.shape)
            padding_shape[1] = self.net.hint_channels - hint.size(1)
            hint = torch.cat(
                [
                    hint,
                    torch.zeros(*padding_shape, dtype=hint.dtype, device=hint.device),
                ],
                dim=1,
            )
        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )

        hint_B_T_H_W_D, _ = self.prepare_hint_embedded_sequence(
            hint, fps=fps, padding_mask=padding_mask
        )
        hint = rearrange(hint_B_T_H_W_D, "B T H W D -> T H W B D")

        guided_hint = self.net.input_hint_block(hint)
        return guided_hint

    def prepare_hint_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_B_T_H_W_D, rope_emb_L_1_1_D = self.net.prepare_hint_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask
        )
        return x_B_T_H_W_D, rope_emb_L_1_1_D

    def prepare_embedded_sequence(
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
            padding_mask = padding_mask.unsqueeze(2).expand(
                x_B_C_T_H_W.size(0), -1, x_B_C_T_H_W.size(2), -1, -1
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask], dim=1
            )  # [B, C+1, T, H, W]

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

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        hint: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        control_weight: Optional[float] = 1.0,
        num_layers_to_use: Optional[int] = -1,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        regional_contexts: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
        base_ratio: float = 0.5,
        **kwargs,
    ):
        guided_hints = self.encode_hint(
            hint, fps=fps, padding_mask=padding_mask, data_type=data_type
        )
        guided_hints = torch.chunk(guided_hints, hint.shape[0] // x.shape[0], dim=3)
        # Only support multi-control at inference time
        assert len(guided_hints) == 1 or not torch.is_grad_enabled()

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = (
            self.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)
        )

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

        if region_masks is None:
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
        else:
            if regional_contexts is None:
                regional_contexts = torch.zeros(self.num_regions, 512, 1, 1024)
                base_ratio = 1.0

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
                regional_contexts=regional_contexts,
                region_masks=region_masks,
                base_ratio=torch.tensor([base_ratio]),
            )

        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
