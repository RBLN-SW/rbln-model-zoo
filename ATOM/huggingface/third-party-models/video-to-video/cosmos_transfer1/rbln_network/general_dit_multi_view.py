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

from typing import Optional, Tuple

import torch
from cosmos_transfer1.diffusion.conditioner import DataType
from cosmos_transfer1.utils import log
from einops import rearrange

from .general_dit import RBLNGeneralDIT


class RBLNMultiViewGeneralDIT(RBLNGeneralDIT):
    def create_runtime(self):
        """
        Creates the RBLN runtime for the model.
        """
        if self._runtime is None:
            if self.rbln_config is None:
                device = None
            else:
                device = self.rbln_config.get("device", None)
            self._runtime = self.compiled_model.create_runtime(
                tensor_type="pt", device=device, timeout=250
            )
        else:
            log.info("Runtime is created already.")

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
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = (
            self.base_model.prepare_embedded_sequence(
                x_B_C_T_H_W,
                fps=fps,
                padding_mask=padding_mask,
                latent_condition=latent_condition,
                latent_condition_sigma=latent_condition_sigma,
                trajectory=trajectory,
                frame_repeat=frame_repeat,
                view_indices_B_T=view_indices_B_T,
            )
        )
        return x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb

    def forward_before_blocks(
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
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
        regional_contexts: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
        """
        trajectory = kwargs.get("trajectory", None)
        frame_repeat = kwargs.get("frame_repeat", None)

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
                trajectory=trajectory,
                frame_repeat=frame_repeat,
                view_indices_B_T=view_indices_B_T,
            )
        )
        # logging affline scale information
        affline_scale_log_info = {}

        timesteps_B_D, adaln_lora_B_3D = self.base_model.t_embedder(timesteps.flatten())
        affline_emb_B_D = timesteps_B_D
        affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

        if scalar_feature is not None:
            raise NotImplementedError("Scalar feature is not implemented yet.")

        affline_scale_log_info["affline_emb_B_D"] = affline_emb_B_D.detach()
        affline_emb_B_D = self.base_model.affline_norm(affline_emb_B_D)

        if self.base_model.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(
                dtype=torch.bool
            )  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        if self.x_format == "THWBD":
            x = rearrange(x_B_T_H_W_D, "B T H W D -> T H W B D")
            if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "B T H W D -> T H W B D"
                )
            crossattn_emb = rearrange(crossattn_emb, "B M D -> M B D")

            if crossattn_mask:
                crossattn_mask = rearrange(crossattn_mask, "B M -> M B")

        elif self.x_format == "BTHWD":
            x = x_B_T_H_W_D
        else:
            raise ValueError(f"Unknown x_format {self.x_format}")
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


class RBLNMultiViewVideoExtendGeneralDIT(RBLNMultiViewGeneralDIT):
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
        video_cond_bool: Optional[torch.Tensor] = None,
        condition_video_indicator: Optional[torch.Tensor] = None,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        condition_video_pose: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Args:
        condition_video_augment_sigma: (B) tensor of sigma value for the conditional
            input augmentation
        condition_video_pose: (B, 1, T, H, W) tensor of pose condition
        """
        B, C, T, H, W = x.shape

        if data_type == DataType.VIDEO:
            assert condition_video_input_mask is not None, (
                "condition_video_input_mask is required for video data type; check "
                "if your model_obj is extend_model.FSDPDiffusionModel or the base "
                "DiffusionModel"
            )
            input_list = [x, condition_video_input_mask]
            if condition_video_pose is not None:
                if condition_video_pose.shape[2] > T:
                    log.warning(
                        f"condition_video_pose has more frames than the input video: "
                        f"{condition_video_pose.shape} > {x.shape}"
                    )
                    condition_video_pose = condition_video_pose[
                        :, :, :T, :, :
                    ].contiguous()
                input_list.append(condition_video_pose)
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
            image_size=image_size,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            condition_video_augment_sigma=condition_video_augment_sigma,
            view_indices_B_T=view_indices_B_T,
            **kwargs,
        )
