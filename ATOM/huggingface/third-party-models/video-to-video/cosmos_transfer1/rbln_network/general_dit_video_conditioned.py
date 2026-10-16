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

from typing import List, Optional

import torch
from cosmos_transfer1.diffusion.conditioner import DataType
from einops import rearrange

from .general_dit import RBLNGeneralDIT


class RBLNVideoExtendGeneralDIT(RBLNGeneralDIT):
    def forward_before_blocks(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        regional_contexts: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks

            condition_video_augment_sigma: (B, T) tensor of sigma value for the
                conditional input augmentation
        """
        del kwargs
        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        original_shape = x.shape
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = (
            self.prepare_embedded_sequence(
                x,
                fps=fps,
                padding_mask=padding_mask,
                latent_condition=latent_condition,
                latent_condition_sigma=latent_condition_sigma,
            )
        )
        # logging affline scale information
        affline_scale_log_info = {}

        timesteps_B_D, adaln_lora_B_3D = self.base_model.t_embedder(timesteps.flatten())
        affline_emb_B_D = timesteps_B_D
        affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

        if scalar_feature is not None:
            raise NotImplementedError("Scalar feature is not implemented yet.")

        if self.base_model.add_augment_sigma_embedding:
            if condition_video_augment_sigma is None:
                # Handling image case
                # Note: for video case, when there is not condition frames, we also
                # set it as zero, see extend_model augment_conditional_latent_frames
                # function
                assert data_type == DataType.IMAGE, (
                    "condition_video_augment_sigma is required for video data type"
                )
                condition_video_augment_sigma = torch.zeros_like(timesteps.flatten())

            affline_augment_sigma_emb_B_D, _ = self.base_model.augment_sigma_embedder(
                condition_video_augment_sigma.flatten()
            )
            affline_emb_B_D = affline_emb_B_D + affline_augment_sigma_emb_B_D
        affline_scale_log_info["affline_emb_B_D"] = affline_emb_B_D.detach()
        affline_emb_B_D = self.base_model.affline_norm(affline_emb_B_D)

        if self.base_model.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(
                dtype=torch.bool
            )  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        x = rearrange(x_B_T_H_W_D, "B T H W D -> T H W B D")
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "B T H W D -> T H W B D"
            )
        crossattn_emb = rearrange(crossattn_emb, "B M D -> M B D")
        if crossattn_mask:
            crossattn_mask = rearrange(crossattn_mask, "B M -> M B")

        # For regional contexts
        if regional_contexts is not None:
            regional_contexts = rearrange(regional_contexts, "B R M D -> R M B D")

        # For region masks (assuming 5D format)
        if region_masks is not None:
            # if len(region_masks.shape) == 5:
            region_masks = rearrange(region_masks, "B R T H W -> R T H W B")

        output = {
            "x": x,
            "affline_emb_B_D": affline_emb_B_D,
            "crossattn_emb": crossattn_emb,
            "crossattn_mask": crossattn_mask,
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_3D": adaln_lora_B_3D,
            "original_shape": original_shape,
            "extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            "regional_contexts": regional_contexts,
            "region_masks": region_masks,
        }
        return output

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
        video_cond_bool: Optional[torch.Tensor] = None,
        condition_video_indicator: Optional[torch.Tensor] = None,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        x_ctrl: Optional[List] = None,
        regional_contexts: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
        base_ratio: float = 0.5,
        **kwargs,
    ):
        if data_type == DataType.VIDEO:
            assert condition_video_input_mask is not None, (
                "condition_video_input_mask is required for video data type"
            )

            input_list = [x, condition_video_input_mask]
            x = torch.cat(
                input_list,
                dim=1,
            )

        return super().forward(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask,
            fps=fps,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            condition_video_augment_sigma=condition_video_augment_sigma,
            x_ctrl=x_ctrl,
            regional_contexts=regional_contexts,
            region_masks=region_masks,
            base_ratio=base_ratio,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
