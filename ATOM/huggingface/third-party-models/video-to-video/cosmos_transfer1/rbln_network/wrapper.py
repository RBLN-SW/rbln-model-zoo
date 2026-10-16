from typing import Optional

import torch
from einops import rearrange


class GeneralDITWrapperWithoutRegion(torch.nn.Module):
    def __init__(self, base_model, original_shape):
        super().__init__()
        self.base_model = base_model
        self.original_shape = original_shape

    def forward(
        self,
        x: torch.Tensor,
        affline_emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        cos_: torch.Tensor,
        sin_: torch.Tensor,
        adaln_lora_B_3D: torch.Tensor,
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D: torch.Tensor,
        *x_ctrl,
    ):
        for i, (name, block) in enumerate(self.base_model.blocks.items()):
            x = block(
                x,
                affline_emb_B_D,
                crossattn_emb,
                crossattn_mask=None,
                rope_emb_L_1_1_D=[cos_, sin_],
                adaln_lora_B_3D=adaln_lora_B_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            if i < len(x_ctrl):
                x = x + x_ctrl[i]

        x_B_T_H_W_D = rearrange(x, "T H W B D -> B T H W D")

        x_B_D_T_H_W = self.base_model.decoder_head(
            x_B_T_H_W_D=x_B_T_H_W_D,
            emb_B_D=affline_emb_B_D,
            crossattn_emb=None,
            origin_shape=self.original_shape,
            crossattn_mask=None,
            adaln_lora_B_3D=adaln_lora_B_3D,
        )

        return x_B_D_T_H_W


class GeneralDITWrapperWithRegion(torch.nn.Module):
    def __init__(self, base_model, original_shape):
        super().__init__()
        self.base_model = base_model
        self.original_shape = original_shape

    def forward(
        self,
        x: torch.Tensor,
        affline_emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        cos_: torch.Tensor,
        sin_: torch.Tensor,
        adaln_lora_B_3D: torch.Tensor,
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D: torch.Tensor,
        regional_contexts: torch.Tensor,
        region_masks: torch.Tensor,
        base_ratio: float,
        *x_ctrl,
    ):
        for i, (name, block) in enumerate(self.base_model.blocks.items()):
            x = block(
                x,
                affline_emb_B_D,
                crossattn_emb,
                crossattn_mask=None,
                rope_emb_L_1_1_D=[cos_, sin_],
                adaln_lora_B_3D=adaln_lora_B_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                regional_contexts=regional_contexts,
                region_masks=region_masks,
                base_ratio=base_ratio,
            )
            if x_ctrl is not None and i < len(x_ctrl):
                x = x + x_ctrl[i]

        x_B_T_H_W_D = rearrange(x, "T H W B D -> B T H W D")

        x_B_D_T_H_W = self.base_model.decoder_head(
            x_B_T_H_W_D=x_B_T_H_W_D,
            emb_B_D=affline_emb_B_D,
            crossattn_emb=None,
            origin_shape=self.original_shape,
            crossattn_mask=None,
            adaln_lora_B_3D=adaln_lora_B_3D,
        )

        return x_B_D_T_H_W


class ControlNetWrapper(torch.nn.Module):
    def __init__(self, controlnet):
        super().__init__()
        self.controlnet = controlnet

    def forward(
        self,
        guided_hints: torch.tensor,
        x: torch.Tensor,
        cos_: torch.Tensor,
        sin_: torch.Tensor,
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        adaln_lora_B_3D: torch.Tensor,
        affline_emb_B_D: torch.Tensor,
        control_weight: torch.Tensor,
        regional_contexts: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
        base_ratio: float = 0.5,
        **kwargs,
    ):
        outs = ()

        num_control_blocks = self.controlnet.layer_mask.index(True)
        num_layers_to_use = num_control_blocks
        control_gate_per_layer = [
            i < num_layers_to_use for i in range(num_control_blocks)
        ]

        for idx, (name, block) in enumerate(self.controlnet.blocks.items()):
            x = block(
                x,
                affline_emb_B_D,
                crossattn_emb,
                crossattn_mask=None,
                rope_emb_L_1_1_D=[cos_, sin_],
                adaln_lora_B_3D=adaln_lora_B_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                regional_contexts=regional_contexts,
                region_masks=region_masks,
                base_ratio=base_ratio,
            )
            if guided_hints is not None:
                x = x + guided_hints
                guided_hints = None

            gate = control_gate_per_layer[idx]
            if isinstance(control_weight, (float, int)) or control_weight.ndim < 2:
                hint_val = self.controlnet.zero_blocks[name](x) * control_weight * gate
            else:
                control_feat = self.controlnet.zero_blocks[name](x)
                weight_map = control_weight
                weight_map = weight_map.permute(2, 3, 4, 0, 1)
                hint_val = control_feat * weight_map * gate
            outs = outs + (hint_val,)

        return outs
