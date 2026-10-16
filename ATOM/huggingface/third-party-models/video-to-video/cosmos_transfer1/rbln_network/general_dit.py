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

from typing import Dict, List, Optional, Tuple

import rebel
import torch
from cosmos_transfer1.diffusion.conditioner import DataType
from cosmos_transfer1.utils import log
from einops import rearrange


class RBLNGeneralDIT:
    """
    An implementation of Cosmos Transfer1's GeneralDIT for RBLN.

    Args:
        compiled_model: The model compiled with RBLN.
        base_model: The original GeneralDIT model.
        rbln_config: The configuration for creating the RBLN runtime.
        num_regions: The number of regions that is used when regional prompt is used.
    """

    def __init__(
        self,
        compiled_model: rebel.RBLNCompiledModel,
        base_model: torch.nn.Module,
        rbln_config: Optional[Dict] = None,
        num_regions: Optional[int] = None,
    ):
        self.compiled_model = compiled_model
        self.base_model = base_model

        self.x_format = self.base_model.blocks["block0"].x_format
        del self.base_model.blocks

        self.rbln_config = rbln_config
        self.num_regions = num_regions
        self._runtime = None

        self.dtype = torch.float32
        self.cp_group = None

    @property
    def runtime(self):
        """
        Returns:
            The RBLN runtime for the model. If the runtime is not created, it will
            be created automatically.
        """
        if self._runtime is None:
            log.info("Runtime is not created. Creating runtime automatically.")
            self.create_runtime()
        return self._runtime

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
                tensor_type="pt", device=device
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and
        handling padding masks.
        """
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = (
            self.base_model.prepare_embedded_sequence(
                x_B_C_T_H_W,
                fps=fps,
                padding_mask=padding_mask,
                latent_condition=latent_condition,
                latent_condition_sigma=latent_condition_sigma,
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

            # For regional contexts
            if regional_contexts is not None:
                regional_contexts = rearrange(regional_contexts, "B R M D -> R M B D")

            # For region masks (assuming 5D format)
            if region_masks is not None:
                region_masks = rearrange(region_masks, "B R T H W -> R T H W B")

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
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        x_ctrl: Optional[List] = None,
        regional_contexts: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
        base_ratio: float = 0.5,
        **kwargs,
    ):
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
            condition_video_augment_sigma: (B,) used in lvg(long video generation),
                we add noise with this sigma to augment condition input, the lvg model
                will condition on the condition_video_augment_sigma value; we need
                forward_before_blocks pass to the forward_before_blocks function.
            regional_contexts: Optional list of regional prompt embeddings, each of shape (B, N, D)
            region_masks: Optional tensor of region masks of shape (B, R, THW)
        """

        inputs = self.forward_before_blocks(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask,
            fps=fps,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
            condition_video_augment_sigma=condition_video_augment_sigma,
            regional_contexts=regional_contexts,
            region_masks=region_masks,
            **kwargs,
        )
        (
            x,
            affline_emb_B_D,
            crossattn_emb,
            crossattn_mask,
            rope_emb_L_1_1_D,
            adaln_lora_B_3D,
            original_shape,
        ) = (
            inputs["x"],
            inputs["affline_emb_B_D"],
            inputs["crossattn_emb"],
            inputs["crossattn_mask"],
            inputs["rope_emb_L_1_1_D"],
            inputs["adaln_lora_B_3D"],
            inputs["original_shape"],
        )
        if rope_emb_L_1_1_D is not None:
            cos_, sin_ = rope_emb_L_1_1_D
        if regional_contexts is not None:
            regional_contexts = inputs["regional_contexts"]
        if region_masks is not None:
            region_masks = inputs["region_masks"]
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = inputs[
            "extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D"
        ]
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape} {original_shape}"
            )

        x_ctrl = {f"x_ctrl_{i}": x_ctrl[i] for i in range(len(x_ctrl))}

        if region_masks is None:
            output = self.runtime(
                x,
                affline_emb_B_D,
                crossattn_emb,
                cos_=cos_,
                sin_=sin_,
                adaln_lora_B_3D=adaln_lora_B_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                **x_ctrl,
            )
        else:
            if regional_contexts is None:
                regional_contexts = torch.zeros(self.num_regions, 512, 1, 1024)
                base_ratio = 1.0
            output = self.runtime(
                x,
                affline_emb_B_D,
                crossattn_emb,
                cos_=cos_,
                sin_=sin_,
                adaln_lora_B_3D=adaln_lora_B_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                regional_contexts=regional_contexts,
                region_masks=region_masks,
                base_ratio=torch.tensor([base_ratio]),
                **x_ctrl,
            )

        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
