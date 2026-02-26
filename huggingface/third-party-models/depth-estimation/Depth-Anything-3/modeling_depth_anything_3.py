# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RBLN-optimized Depth Anything 3 for depth estimation on RBLN NPU devices.
"""

from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import rebel
import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict as AttrDict
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.model.dualdpt import DualDPT
from depth_anything_3.model.utils.head_utils import custom_interpolate
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.utils.geometry import affine_inverse
from einops import rearrange
from optimum.rbln import RBLNCompileConfig, RBLNModel, RBLNModelConfig
from optimum.rbln.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
    )


def _patched_dualdpt_forward(
    self,
    feats: List[torch.Tensor],
    H: int,
    W: int,
    patch_start_idx: int,
) -> dict:
    """DualDPT forward using f[..., -1:].squeeze(-1) for conf to avoid RBLN-incompatible take."""
    B, _, C = feats[0].shape
    ph, pw = H // self.patch_size, W // self.patch_size
    resized = []
    for i, k in enumerate(self.intermediate_layer_idx):
        x = (
            self.norm(feats[k][:, patch_start_idx:])
            .permute(0, 2, 1)
            .reshape(B, C, ph, pw)
        )
        x = self.projects[i](x)
        if self.pos_embed:
            x = self._add_pos_embed(x, W, H)
        resized.append(self.resize_layers[i](x))

    fused_main, fused_aux_pyr = self._fuse(resized)
    h, w = (
        int(ph * self.patch_size / self.down_ratio),
        int(pw * self.patch_size / self.down_ratio),
    )
    fused_main = custom_interpolate(
        fused_main, (h, w), mode="bilinear", align_corners=True
    )
    if self.pos_embed:
        fused_main = self._add_pos_embed(fused_main, W, H)

    def apply_conf_channel(f: torch.Tensor) -> torch.Tensor:
        return self._apply_activation_single(
            f[..., -1:].squeeze(-1), self.conf_activation
        )

    fmap = self.scratch.output_conv2(fused_main).permute(0, 2, 3, 1)
    main_pred = self._apply_activation_single(fmap[..., :-1], self.activation)
    main_conf = apply_conf_channel(fmap)

    last_aux = fused_aux_pyr[-1]
    if self.pos_embed:
        last_aux = self._add_pos_embed(last_aux, W, H)
    fmap_last = self.scratch.output_conv2_aux[-1](last_aux).permute(0, 2, 3, 1)
    aux_pred = self._apply_activation_single(fmap_last[..., :-1], "linear")
    aux_conf = apply_conf_channel(fmap_last)

    return {
        self.head_main: main_pred.squeeze(-1),
        f"{self.head_main}_conf": main_conf,
        self.head_aux: aux_pred,
        f"{self.head_aux}_conf": aux_conf,
    }


DualDPT._forward_impl = _patched_dualdpt_forward


class DinoV2BackboneWrapper(nn.Module):
    """Wraps DinoV2 ViT with precomputed RoPE and key_valid masking for RBLN compile.

    forward returns exactly 4 layer outputs (feat0..feat3) plus cam_token_out.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        norm: nn.Module,
        embed_dim: int,
        num_heads: int,
        out_layers: List[int],
        alt_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        num_register_tokens: int = 0,
        camera_token: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.blocks = blocks
        self.norm = norm
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.out_layers = out_layers
        self.alt_start = alt_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_register_tokens = num_register_tokens

        if camera_token is not None:
            self.camera_token = nn.Parameter(camera_token.clone())
        else:
            self.camera_token = None

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half dims: [x1, x2] -> [-x2, x1] for RoPE."""
        feature_dim = x.shape[-1]
        x1 = x[..., : feature_dim // 2]
        x2 = x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope_precomputed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D RoPE (cos/sin for y and x) to q and k; returns (q_embed, k_embed)."""
        head_dim = q.shape[-1]
        half_dim = head_dim // 2

        q_v, q_h = q[..., :half_dim], q[..., half_dim:]
        k_v, k_h = k[..., :half_dim], k[..., half_dim:]

        cos_y, cos_x = cos[..., :half_dim], cos[..., half_dim:]
        sin_y, sin_x = sin[..., :half_dim], sin[..., half_dim:]

        q_v_embed = (q_v * cos_y) + (self.rotate_half(q_v) * sin_y)
        k_v_embed = (k_v * cos_y) + (self.rotate_half(k_v) * sin_y)

        q_h_embed = (q_h * cos_x) + (self.rotate_half(q_h) * sin_x)
        k_h_embed = (k_h * cos_x) + (self.rotate_half(k_h) * sin_x)

        q_embed = torch.cat([q_v_embed, q_h_embed], dim=-1)
        k_embed = torch.cat([k_v_embed, k_h_embed], dim=-1)

        return q_embed, k_embed

    def attention_with_precomputed_rope(
        self,
        blk: nn.Module,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        key_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One block: norm->qkv, optional RoPE on q/k, key_valid mask, SDPA, residual."""
        x_norm = blk.norm1(x)
        B, N, C = x_norm.shape
        qkv = (
            blk.attn.qkv(x_norm)
            .reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = blk.attn.q_norm(q), blk.attn.k_norm(k)

        if cos is not None and sin is not None:
            q, k = self.apply_rope_precomputed(q, k, cos, sin)

        attn_bias = None
        if key_valid is not None:
            attn_bias = (1.0 - key_valid).to(dtype=q.dtype) * (-1.0e4)
            attn_bias = attn_bias[:, None, None, :]

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            attn_mask=attn_bias,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = blk.attn.proj(attn_out)
        attn_out = blk.attn.proj_drop(attn_out)

        x = x + blk.ls1(attn_out)
        x = x + blk.ls2(blk.mlp(blk.norm2(x)))

        return x

    def forward(
        self,
        x: torch.Tensor,
        rope_cos_local: torch.Tensor,
        rope_sin_local: torch.Tensor,
        rope_cos_global: torch.Tensor,
        rope_sin_global: torch.Tensor,
        key_valid_local: Optional[torch.Tensor] = None,
        key_valid_global: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run ViT blocks with local/global RoPE and key_valid; return 4 layer feats + cam_token_out."""
        B, S, N, C = x.shape
        outputs = []
        cam_token_out: Optional[torch.Tensor] = None
        local_x = x
        max_out_layer = max(self.out_layers)

        for i, blk in enumerate(self.blocks):
            if self.alt_start != -1 and i == self.alt_start:
                x = x.clone()
                if self.camera_token is not None:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    learned_cam_token = torch.cat([ref_token, src_token], dim=1)
                    x[:, :, 0] = learned_cam_token

            use_rope = (i >= self.rope_start) and (self.rope_start != -1)
            is_global = self.alt_start != -1 and i >= self.alt_start and i % 2 == 1
            if is_global:
                x_flat = rearrange(x, "b s n c -> b (s n) c")
                cos = rope_cos_global if use_rope else None
                sin = rope_sin_global if use_rope else None
                key_valid = key_valid_global
                x_flat = self.attention_with_precomputed_rope(
                    blk, x_flat, cos, sin, key_valid=key_valid
                )
                x = rearrange(x_flat, "b (s n) c -> b s n c", s=S)
            else:
                x_flat = rearrange(x, "b s n c -> (b s) n c")
                cos = rope_cos_local if use_rope else None
                sin = rope_sin_local if use_rope else None
                key_valid = key_valid_local
                x_flat = self.attention_with_precomputed_rope(
                    blk, x_flat, cos, sin, key_valid=key_valid
                )
                x = rearrange(x_flat, "(b s) n c -> b s n c", b=B, s=S)
                local_x = x

            if i in self.out_layers:
                if self.cat_token:
                    out_x = torch.cat([local_x, x], dim=-1)
                else:
                    out_x = x
                outputs.append(out_x)
                if i == max_out_layer:
                    cam_token_out = out_x[:, :, 0]

        processed_outputs = []
        for out in outputs:
            if out.shape[-1] == self.embed_dim:
                out = self.norm(out)
            elif out.shape[-1] == self.embed_dim * 2:
                out = torch.cat(
                    [out[..., : self.embed_dim], self.norm(out[..., self.embed_dim :])],
                    dim=-1,
                )
            out = out[..., 1 + self.num_register_tokens :, :]
            processed_outputs.append(out)

        if cam_token_out is None:
            cam_token_out = x[:, :, 0]

        return (
            processed_outputs[0],
            processed_outputs[1],
            processed_outputs[2],
            processed_outputs[3],
            cam_token_out,
        )


class DualDPTHeadWrapper(nn.Module):
    """Thin wrapper around DualDPT head; set_image_size must be called before compile."""

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head
        self.image_size: Optional[Tuple[int, int]] = None

    def set_image_size(self, image_size: Tuple[int, int]) -> None:
        """Set (H, W) for head forward; required before compile."""
        self.image_size = tuple(image_size)

    def forward(
        self,
        feat0: torch.Tensor,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        feat3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run DualDPT head; returns (depth, depth_conf, ray, ray_conf)."""
        if self.image_size is None:
            raise RuntimeError("`set_image_size()` must be called before compile.")

        H, W = self.image_size
        feats = [(feat0, None), (feat1, None), (feat2, None), (feat3, None)]
        output = self.head(feats, H, W, patch_start_idx=0)

        depth = output.depth
        depth_conf = output.depth_conf
        ray = output.get("ray", torch.zeros(1, device=depth.device))
        ray_conf = output.get("ray_conf", torch.zeros(1, device=depth.device))

        return depth, depth_conf, ray, ray_conf


class CameraDecWrapper(nn.Module):
    """Passthrough wrapper for camera decoder (cam_token -> pose encoding)."""

    def __init__(self, cam_dec: nn.Module):
        super().__init__()
        self.cam_dec = cam_dec

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Map cam_token features to pose encoding."""
        return self.cam_dec(feat)


class RoPEPrecomputer:
    """Precomputes RoPE cos/sin for local and global attention (2D patch positions)."""

    def __init__(
        self,
        head_dim: int,
        patch_size: int,
        num_register_tokens: int = 0,
        base_frequency: float = 100.0,
        patch_start_idx: int = 1,
    ):
        self.head_dim = head_dim
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.base_frequency = base_frequency
        self.feature_dim = head_dim // 2
        self.patch_start_idx = patch_start_idx

    def _compute_freqs(
        self, max_pos: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin tables for positions [0, max_pos); returns (cos, sin) of shape (max_pos, feature_dim)."""
        exponents = (
            torch.arange(0, self.feature_dim, 2, device=device).float()
            / self.feature_dim
        )
        inv_freq = 1.0 / (self.base_frequency**exponents)
        positions = torch.arange(max_pos, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def _get_2d_positions(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return (ph*pw, 2) tensor of (y, x) patch indices in row-major order."""
        ph, pw = H // self.patch_size, W // self.patch_size
        y_coords = torch.arange(ph, device=device)
        x_coords = torch.arange(pw, device=device)
        positions = torch.cartesian_prod(y_coords, x_coords)
        return positions

    def _expand_local_rope(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        B: int,
        S: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand (seq_len, head_dim) cos/sin to (B*S, 1, seq_len, head_dim)."""
        expanded_cos = (
            cos.unsqueeze(0).unsqueeze(0).expand(B * S, 1, seq_len, -1).clone()
        )
        expanded_sin = (
            sin.unsqueeze(0).unsqueeze(0).expand(B * S, 1, seq_len, -1).clone()
        )
        return expanded_cos, expanded_sin

    def _expand_global_rope(
        self,
        cos_per_view: torch.Tensor,
        sin_per_view: torch.Tensor,
        B: int,
        S: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand per-view (seq_len, head_dim) to (B, 1, S*seq_len, head_dim)."""
        cos_flat = cos_per_view.repeat(S, 1)
        sin_flat = sin_per_view.repeat(S, 1)
        expanded_cos = (
            cos_flat.unsqueeze(0).unsqueeze(0).expand(B, 1, S * seq_len, -1).clone()
        )
        expanded_sin = (
            sin_flat.unsqueeze(0).unsqueeze(0).expand(B, 1, S * seq_len, -1).clone()
        )
        return expanded_cos, expanded_sin

    def compute(
        self, B: int, S: int, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (rope_cos_local, rope_sin_local, rope_cos_global, rope_sin_global) for (B, S, H, W)."""
        ph, pw = H // self.patch_size, W // self.patch_size
        num_patches = ph * pw
        seq_len = 1 + self.num_register_tokens + num_patches

        patch_positions = self._get_2d_positions(H, W, device) + 1

        special_positions = torch.zeros(
            self.patch_start_idx, 2, device=device, dtype=torch.long
        )
        if self.num_register_tokens > 0:
            full_positions = torch.cat(
                [
                    special_positions,
                    patch_positions[: self.num_register_tokens],
                    patch_positions,
                ],
                dim=0,
            )
        else:
            full_positions = torch.cat([special_positions, patch_positions], dim=0)
        pos_y = full_positions[:, 0]
        pos_x = full_positions[:, 1]

        max_pos = max(ph, pw) + 2
        cos_table, sin_table = self._compute_freqs(max_pos, device, dtype)

        cos_y = cos_table[pos_y]
        sin_y = sin_table[pos_y]
        cos_x = cos_table[pos_x]
        sin_x = sin_table[pos_x]

        cos_local = torch.cat([cos_y, cos_x], dim=-1)
        sin_local = torch.cat([sin_y, sin_x], dim=-1)
        rope_cos_local, rope_sin_local = self._expand_local_rope(
            cos_local, sin_local, B, S, seq_len
        )

        global_positions = torch.ones(seq_len, device=device, dtype=torch.long)
        global_positions[: self.patch_start_idx] = 0
        cos_g = cos_table[global_positions]
        sin_g = sin_table[global_positions]
        cos_global_per_view = torch.cat([cos_g, cos_g], dim=-1)
        sin_global_per_view = torch.cat([sin_g, sin_g], dim=-1)
        rope_cos_global, rope_sin_global = self._expand_global_rope(
            cos_global_per_view, sin_global_per_view, B, S, seq_len
        )

        return rope_cos_local, rope_sin_local, rope_cos_global, rope_sin_global


def pad_to_static_size(
    x: torch.Tensor, target_h: int, target_w: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad input (B, S, C, H, W) to (target_h, target_w); return (padded, (H_orig, W_orig))."""
    B, S, _, H, W = x.shape

    if H == target_h and W == target_w:
        return x, (H, W)

    pad_h = target_h - H
    pad_w = target_w - W

    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"Input size ({H}, {W}) must not exceed target ({target_h}, {target_w})."
        )

    x = rearrange(x, "b s c h w -> (b s) c h w")
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    x = rearrange(x, "(b s) c h w -> b s c h w", b=B, s=S)

    return x, (H, W)


def _replicate_edge_invalid_patches(
    feat: torch.Tensor,
    ph_orig: int,
    pw_orig: int,
    ph_static: int,
    pw_static: int,
) -> torch.Tensor:
    """Fill invalid (padded) patches by replicating the last valid row/col edge.

    Zeroing creates a sharp boundary that corrupts conf at the pad edge (head conv
    receptive field spans valid+invalid). Replicating the edge avoids this.
    """
    if ph_orig <= 0 or pw_orig <= 0 or (ph_orig >= ph_static and pw_orig >= pw_static):
        return feat

    B, S, N, C = feat.shape
    feat_2d = feat.reshape(B, S, ph_static, pw_static, C)

    if ph_orig < ph_static:
        last_row = feat_2d[:, :, ph_orig - 1 : ph_orig, :pw_orig, :]
        feat_2d[:, :, ph_orig:, :pw_orig, :] = last_row.expand(
            B, S, ph_static - ph_orig, pw_orig, C
        )
        if pw_orig < pw_static:
            corner = feat_2d[:, :, ph_orig - 1 : ph_orig, pw_orig - 1 : pw_orig, :]
            feat_2d[:, :, ph_orig:, pw_orig:, :] = corner.expand(
                B, S, ph_static - ph_orig, pw_static - pw_orig, C
            )
    if pw_orig < pw_static:
        last_col = feat_2d[:, :, :ph_orig, pw_orig - 1 : pw_orig, :]
        feat_2d[:, :, :ph_orig, pw_orig:, :] = last_col.expand(
            B, S, ph_orig, pw_static - pw_orig, C
        )

    return feat_2d.reshape(B, S, N, C)


def _replicate_edge_feats(
    feat0: torch.Tensor,
    feat1: torch.Tensor,
    feat2: torch.Tensor,
    feat3: torch.Tensor,
    ph_orig: int,
    pw_orig: int,
    ph_static: int,
    pw_static: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply edge replication to four backbone feats when padded; no-op if not padded."""
    if ph_orig >= ph_static and pw_orig >= pw_static:
        return feat0, feat1, feat2, feat3
    return tuple(
        _replicate_edge_invalid_patches(f, ph_orig, pw_orig, ph_static, pw_static)
        for f in (feat0, feat1, feat2, feat3)
    )


def depad_output(
    output: AttrDict,
    original_hw: Tuple[int, int],
    down_ratio: int = 1,
    patch_size: int = 14,
) -> AttrDict:
    """Crop output to valid patch region (excludes padded content)."""
    orig_h, orig_w = original_hw
    ph_orig, pw_orig = orig_h // patch_size, orig_w // patch_size
    out_h = (ph_orig * patch_size) // down_ratio
    out_w = (pw_orig * patch_size) // down_ratio

    def _crop_if_needed(tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None or tensor.numel() <= 1:
            return tensor
        actual_h, actual_w = tensor.shape[-2], tensor.shape[-1]
        if actual_h > out_h or actual_w > out_w:
            return tensor[..., :out_h, :out_w].contiguous()
        return tensor

    for key in ("depth", "depth_conf", "ray", "ray_conf"):
        if key not in output:
            continue
        tensor = output[key]
        if tensor is None:
            continue
        if key in ("ray", "ray_conf") and tensor.numel() <= 1:
            continue
        output[key] = _crop_if_needed(tensor)

    return output


def _normalize_image_size(
    image_size: Optional[Union[int, List[int], Tuple[int, int]]],
) -> Tuple[int, int]:
    """Return (H, W) from int, [H, W], or (H, W); default (504, 504). Idempotent when already (H, W).

    Raises:
        ValueError: If `image_size` is not int, (H, W), or positive.
    """
    if image_size is None:
        return (504, 504)
    if isinstance(image_size, int):
        return (image_size, image_size)
    if isinstance(image_size, list):
        image_size = tuple(image_size)

    if not isinstance(image_size, tuple) or len(image_size) != 2:
        raise ValueError(f"`image_size` must be an int or (H, W), got {image_size!r}.")

    h, w = image_size
    if not isinstance(h, Integral) or not isinstance(w, Integral):
        raise ValueError(
            f"`image_size` elements must be integers, got {(type(h).__name__, type(w).__name__)}."
        )
    h, w = int(h), int(w)
    if h <= 0 or w <= 0:
        raise ValueError(f"`image_size` must be positive, got {(h, w)}.")

    return (h, w)


class _TokenBuilder(nn.Module):
    """Builds patch tokens + pos_embed; matches DinoVisionTransformer.interpolate_pos_encoding / prepare_tokens_with_masks."""

    def __init__(
        self,
        patch_embed: nn.Module,
        pos_embed: torch.Tensor,
        cls_token: torch.Tensor,
        register_tokens: Optional[torch.Tensor],
        patch_size: int,
        interpolate_antialias: bool,
        interpolate_offset: float,
        embed_dim: int,
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.register_buffer("_pos_embed", pos_embed)
        self.register_buffer("_cls_token", cls_token)
        self._has_register_tokens = register_tokens is not None
        reg = (
            register_tokens
            if register_tokens is not None
            else torch.empty(
                0, embed_dim, device=cls_token.device, dtype=cls_token.dtype
            )
        )
        self.register_buffer("_register_tokens", reg)
        self.patch_size = patch_size
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.embed_dim = embed_dim

    def interpolate_pos_encoding(
        self, x_tokens: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """Interpolate stored patch pos_embed to (H, W) patch grid; preserves class token."""
        previous_dtype = x_tokens.dtype
        npatch = x_tokens.shape[1] - 1
        N = self._pos_embed.shape[1] - 1
        if npatch == N and H == W:
            return self._pos_embed.to(previous_dtype)
        pos_embed = self._pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x_tokens.shape[-1]
        ph = H // self.patch_size
        pw = W // self.patch_size
        M = int(math.sqrt(N))
        if N != M * M:
            raise ValueError(
                f"pos_embed patch count must be a perfect square, got {N}."
            )
        kwargs = {}
        if self.interpolate_offset:
            scale_y = float(ph + self.interpolate_offset) / M
            scale_x = float(pw + self.interpolate_offset) / M
            kwargs["scale_factor"] = (scale_y, scale_x)
        else:
            kwargs["size"] = (ph, pw)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        if (ph, pw) != tuple(patch_pos_embed.shape[-2:]):
            raise ValueError(
                f"Interpolated pos_embed shape must be (ph, pw)={(ph, pw)}, got {tuple(patch_pos_embed.shape[-2:])}."
            )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens_with_pos(self, x: torch.Tensor) -> torch.Tensor:
        """Build patch tokens + cls + register, add pos_embed; output (B, S, N, C)."""
        B, S, _, H, W = x.shape
        x_flat = rearrange(x, "b s c h w -> (b s) c h w")
        patch_tokens = self.patch_embed(x_flat)
        cls_token = self._cls_token.expand(B * S, -1, -1)
        tokens = torch.cat((cls_token, patch_tokens), dim=1)
        tokens = tokens + self.interpolate_pos_encoding(tokens, H, W)
        if self._has_register_tokens:
            reg = self._register_tokens.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((tokens[:, :1], reg, tokens[:, 1:]), dim=1)
        tokens = rearrange(tokens, "(b s) n c -> b s n c", b=B, s=S)
        return tokens


class RBLNDepthAnything3Config(RBLNModelConfig):
    """
    Configuration class for RBLNDepthAnything3.

    This configuration class stores the configuration parameters for RBLN-compiled
    Depth Anything 3 (batch_size, num_images, image_size). PATCH_SIZE is taken from
    DepthAnything3Net.
    """

    PATCH_SIZE = DepthAnything3Net.PATCH_SIZE

    def __init__(
        self,
        batch_size: int = 1,
        num_images: int = 1,
        image_size: Optional[Tuple[int, int]] = None,
        use_ray_pose: bool = False,
        **kwargs,
    ):
        """
        Args:
            batch_size (`int`, *optional*, defaults to `1`): Batch size for compile.
            num_images (`int`, *optional*, defaults to `2`): Number of images per sample.
            image_size (`tuple` of (`int`, `int`) or `int`, *optional*): Static (H, W) or single int for square; default (504, 504).
            use_ray_pose (`bool`, *optional*, defaults to `False`): If `True`, pose will use ray (cam_dec not compiled, no cam_dec.rbln); if `False`, cam_dec is compiled for pose (cam_dec.rbln).
            **kwargs: Passed to [`RBLNModelConfig`].
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_images = num_images
        self.image_size = image_size
        self.use_ray_pose = use_ray_pose


class RBLNDepthAnything3(RBLNModel):
    """
    RBLN optimized Depth Anything 3 model for depth estimation.

    This class provides hardware-accelerated inference for Depth Anything 3 on RBLN devices,
    with static-shape compilation, RoPE/key_valid support, and optional camera decoder.
    Use RBLNDepthAnything3Config as rbln_config in from_pretrained or from_model.
    """

    _BACKBONE_MODEL_NAME = "dinov2"
    _HEAD_MODEL_NAME = "head"
    _CAM_DEC_MODEL_NAME = "cam_dec"

    def _build_tokens_and_key_valid(
        self,
        images: torch.Tensor,
        original_hw: Tuple[int, int],
        padded_hw: Tuple[int, int],
        num_special: int,
        ph_static: int,
        pw_static: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Build static-shape tokens with valid-region mask; return tokens, key_valid_* , ph_orig, pw_orig."""
        B, S, _, _, _ = images.shape
        H_orig, W_orig = original_hw
        H_pad, W_pad = padded_hw
        patch_size = RBLNDepthAnything3Config.PATCH_SIZE
        ph_orig = H_orig // patch_size
        pw_orig = W_orig // patch_size
        static_num_patches = ph_static * pw_static

        tokens_small = self.token_builder.prepare_tokens_with_pos(images)
        embed_dim = tokens_small.shape[-1]
        tokens = torch.zeros(
            (B, S, num_special + static_num_patches, embed_dim),
            device=tokens_small.device,
            dtype=tokens_small.dtype,
        )
        tokens[:, :, :num_special, :] = tokens_small[:, :, :num_special, :]

        patch_tokens_all = tokens_small[:, :, num_special:, :].reshape(
            B, S, ph_static, pw_static, embed_dim
        )
        patch_tokens = patch_tokens_all[:, :, :ph_orig, :pw_orig, :]
        patch_tokens = self._maybe_realign_patch_tokens(
            patch_tokens=patch_tokens,
            embed_dim=embed_dim,
            original_hw=(H_orig, W_orig),
            padded_hw=(H_pad, W_pad),
            patch_hw=(ph_orig, pw_orig),
        )

        patch_static = torch.zeros(
            (B, S, ph_static, pw_static, embed_dim),
            device=tokens_small.device,
            dtype=tokens_small.dtype,
        )
        patch_static[:, :, :ph_orig, :pw_orig, :] = patch_tokens
        tokens[:, :, num_special:, :] = patch_static.reshape(
            B, S, static_num_patches, embed_dim
        )

        key_valid_per_view = self._build_key_valid_per_view(
            num_special=num_special,
            ph_orig=ph_orig,
            pw_orig=pw_orig,
            ph_static=ph_static,
            pw_static=pw_static,
            device=tokens_small.device,
        )
        key_valid_expanded = key_valid_per_view[None, None, :].expand(B, S, -1)
        key_valid_local = key_valid_expanded.reshape(-1, key_valid_expanded.shape[-1])
        key_valid_global = key_valid_expanded.reshape(B, -1)
        return tokens, key_valid_local, key_valid_global, ph_orig, pw_orig

    def _build_key_valid_per_view(
        self,
        num_special: int,
        ph_orig: int,
        pw_orig: int,
        ph_static: int,
        pw_static: int,
        device: torch.device,
    ) -> torch.Tensor:
        """One view mask: [1 for special tokens] + [1 for valid patches, 0 for pad]; shape (num_special + ph_static*pw_static,)."""
        patch_valid = torch.zeros(
            (ph_static, pw_static), device=device, dtype=torch.float32
        )
        patch_valid[:ph_orig, :pw_orig] = 1.0
        return torch.cat(
            [
                torch.ones((num_special,), device=device, dtype=torch.float32),
                patch_valid.reshape(-1),
            ],
            dim=0,
        )

    def _maybe_realign_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        embed_dim: int,
        original_hw: Tuple[int, int],
        padded_hw: Tuple[int, int],
        patch_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """If padded, realign patch positions: subtract padded pos_embed and add original pos_embed."""
        H_orig, W_orig = original_hw
        H_pad, W_pad = padded_hw
        ph_orig, pw_orig = patch_hw
        if (H_orig, W_orig) == (H_pad, W_pad):
            return patch_tokens

        device, dtype = patch_tokens.device, patch_tokens.dtype
        patch_size = self.token_builder.patch_size
        ph_full, pw_full = H_pad // patch_size, W_pad // patch_size
        dummy = torch.zeros(
            1, 1 + ph_full * pw_full, embed_dim, device=device, dtype=dtype
        )
        pos_padded = self.token_builder.interpolate_pos_encoding(dummy, H_pad, W_pad)
        pos_padded = pos_padded[:, 1:, :].reshape(1, ph_full, pw_full, embed_dim)
        pos_padded = pos_padded[:, :ph_orig, :pw_orig, :].unsqueeze(1)

        dummy_orig = torch.zeros(
            1, 1 + ph_orig * pw_orig, embed_dim, device=device, dtype=dtype
        )
        pos_orig = self.token_builder.interpolate_pos_encoding(
            dummy_orig, H_orig, W_orig
        )
        pos_orig = pos_orig[:, 1:, :].reshape(1, 1, ph_orig, pw_orig, embed_dim)
        return patch_tokens - pos_padded + pos_orig

    def _populate_pose_outputs(
        self,
        output: AttrDict,
        ray: Optional[torch.Tensor],
        ray_conf: Optional[torch.Tensor],
        cam_token_out: torch.Tensor,
        original_hw: Tuple[int, int],
        use_ray_pose: bool = False,
    ) -> None:
        """Set output.extrinsics and output.intrinsics. use_ray_pose=True -> ray only; False -> cam_dec only. Ray is produced by the head (DualDPT), not a separate module."""
        if use_ray_pose and ray is not None and ray.numel() > 1:
            from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray

            pred_extrinsic, pred_focal_lengths, pred_principal_points = (
                get_extrinsic_from_camray(
                    ray,
                    ray_conf,
                    ray.shape[-3],
                    ray.shape[-2],
                )
            )
            pred_extrinsic = affine_inverse(pred_extrinsic)[:, :, :3, :]
            orig_w, orig_h = original_hw[1], original_hw[0]
            fx = pred_focal_lengths[:, :, 0] / 2 * orig_w
            fy = pred_focal_lengths[:, :, 1] / 2 * orig_h
            cx = pred_principal_points[:, :, 0] * orig_w * 0.5
            cy = pred_principal_points[:, :, 1] * orig_h * 0.5
            batch_pose, num_views_pose = (
                pred_extrinsic.shape[0],
                pred_extrinsic.shape[1],
            )
            pred_intrinsic = (
                torch.eye(3, device=pred_extrinsic.device)[None, None]
                .expand(batch_pose, num_views_pose, 3, 3)
                .clone()
            )
            pred_intrinsic[:, :, 0, 0] = fx
            pred_intrinsic[:, :, 1, 1] = fy
            pred_intrinsic[:, :, 0, 2] = cx
            pred_intrinsic[:, :, 1, 2] = cy

            output.extrinsics = pred_extrinsic
            output.intrinsics = pred_intrinsic
            return

        if not use_ray_pose and self.cam_dec_runtime is not None:
            pose_enc = self.cam_dec_runtime(cam_token_out)
            c2w, intrinsics = pose_encoding_to_extri_intri(pose_enc, original_hw)
            output.extrinsics = affine_inverse(c2w)[:, :, :3, :]
            output.intrinsics = intrinsics

    @staticmethod
    def _build_compile_cfg_map(
        compile_cfgs: List[RBLNCompileConfig],
    ) -> dict[str, RBLNCompileConfig]:
        """Map compiled_model_name -> config; raises if duplicate names."""
        cfg_map = {cfg.compiled_model_name: cfg for cfg in compile_cfgs}
        if len(cfg_map) != len(compile_cfgs):
            raise ValueError("Duplicate `compiled_model_name` in compile_cfgs.")
        return cfg_map

    def _warn_unsupported_camera_inputs(
        self,
        extrinsics: Optional[torch.Tensor],
        intrinsics: Optional[torch.Tensor],
    ) -> None:
        """Log warning when user passes extrinsics/intrinsics (learned camera tokens are used)."""
        for name, value in [("extrinsics", extrinsics), ("intrinsics", intrinsics)]:
            if value is not None:
                logger.warning(
                    f"`{name}` provided but external camera tokens are not supported. "
                    "Using learned camera tokens instead."
                )

    def _assign_runtimes(self) -> None:
        """Set dinov2_runtime, head_runtime, cam_dec_runtime from rbln_config.compile_cfgs order."""
        for i, cfg in enumerate(self.rbln_config.compile_cfgs):
            name = cfg.compiled_model_name
            if name == self._BACKBONE_MODEL_NAME:
                self.dinov2_runtime = self.model[i]
            elif name == self._HEAD_MODEL_NAME:
                self.head_runtime = self.model[i]
            elif name == self._CAM_DEC_MODEL_NAME:
                self.cam_dec_runtime = self.model[i]

    def _run_backbone(
        self,
        tokens: torch.Tensor,
        rope_cos_local: torch.Tensor,
        rope_sin_local: torch.Tensor,
        rope_cos_global: torch.Tensor,
        rope_sin_global: torch.Tensor,
        key_valid_local: torch.Tensor,
        key_valid_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run ViT backbone runtime; returns (feat0, feat1, feat2, feat3, cam_token_out)."""
        return self.dinov2_runtime(
            tokens,
            rope_cos_local,
            rope_sin_local,
            rope_cos_global,
            rope_sin_global,
            key_valid_local,
            key_valid_global,
        )

    @classmethod
    def get_pytorch_model(
        cls,
        model_id: str,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        rbln_config: Optional[RBLNModelConfig] = None,
        **kwargs,
    ) -> "PreTrainedModel":
        """
        Load the HuggingFace DepthAnything3 PyTorch model.

        Args:
            model_id (`str`): HuggingFace model id or local path.
            force_download (`bool`, *optional*, defaults to `False`): Whether to force download.
            cache_dir (`str` or `os.PathLike`, *optional*): Directory to cache downloads.
            subfolder (`str`, *optional*, defaults to `""`): Subfolder in the repo.
            local_files_only (`bool`, *optional*, defaults to `False`): Use only local files.
            rbln_config ([`RBLNModelConfig`], *optional*): Unused for load; for API compatibility.
            **kwargs: Passed through to `DepthAnything3.from_pretrained`.

        Returns:
            [`DepthAnything3`]: Model in eval mode with float32 dtype.
        """
        from transformers import PretrainedConfig

        model = DepthAnything3.from_pretrained(model_id, **kwargs)
        model.dtype = torch.float32
        model.eval()

        model.config = PretrainedConfig(
            model_type="depth_anything_3",
            model_name=model.model_name,
        )

        return model

    @classmethod
    def _wrap_model_if_needed(
        cls, model: "DepthAnything3", rbln_config: RBLNDepthAnything3Config
    ) -> nn.Module:
        """Wrap DinoV2 backbone for compile: RoPE + key_valid and optional camera token."""
        backbone = model.model.backbone
        vit = backbone.pretrained
        camera_token = getattr(vit, "camera_token", None)

        return DinoV2BackboneWrapper(
            blocks=vit.blocks,
            norm=vit.norm,
            embed_dim=vit.embed_dim,
            num_heads=vit.num_heads,
            out_layers=backbone.out_layers,
            alt_start=backbone.alt_start,
            rope_start=backbone.rope_start,
            cat_token=backbone.cat_token,
            num_register_tokens=vit.num_register_tokens,
            camera_token=camera_token,
        )

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[
            Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]
        ] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNDepthAnything3Config] = None,
    ) -> RBLNDepthAnything3Config:
        """Set compile_cfgs (backbone, head, optional cam_dec) from model and rbln_config.

        Raises:
            ValueError: If `model` is `None`, or `image_size` is not divisible by `PATCH_SIZE`.
        """
        if rbln_config is None:
            rbln_config = RBLNDepthAnything3Config()
        if model is None:
            raise ValueError("`model` is required for _update_rbln_config.")

        batch_size = rbln_config.batch_size
        num_images = rbln_config.num_images

        rbln_config.image_size = _normalize_image_size(rbln_config.image_size)
        H, W = rbln_config.image_size
        patch_size = RBLNDepthAnything3Config.PATCH_SIZE
        if (H % patch_size) != 0 or (W % patch_size) != 0:
            raise ValueError(
                f"`image_size` must be divisible by PATCH_SIZE ({patch_size}), got {(H, W)}."
            )

        backbone = model.model.backbone
        vit = backbone.pretrained
        embed_dim = vit.embed_dim
        num_heads = vit.num_heads
        head_dim = embed_dim // num_heads

        num_patches = (H // patch_size) * (W // patch_size)
        num_register_tokens = vit.num_register_tokens
        seq_len = 1 + num_register_tokens + num_patches

        if backbone.cat_token:
            head_feat_dim = embed_dim * 2
        else:
            head_feat_dim = embed_dim

        head_feat_shape = [batch_size, num_images, num_patches, head_feat_dim]
        head_input_info = [
            (name, head_feat_shape, "float32")
            for name in ("feat0", "feat1", "feat2", "feat3")
        ]
        backbone_input_info = [
            ("x", [batch_size, num_images, seq_len, embed_dim], "float32"),
            (
                "rope_cos_local",
                [batch_size * num_images, 1, seq_len, head_dim],
                "float32",
            ),
            (
                "rope_sin_local",
                [batch_size * num_images, 1, seq_len, head_dim],
                "float32",
            ),
            (
                "rope_cos_global",
                [batch_size, 1, num_images * seq_len, head_dim],
                "float32",
            ),
            (
                "rope_sin_global",
                [batch_size, 1, num_images * seq_len, head_dim],
                "float32",
            ),
            ("key_valid_local", [batch_size * num_images, seq_len], "float32"),
            ("key_valid_global", [batch_size, num_images * seq_len], "float32"),
        ]
        compile_cfgs = [
            RBLNCompileConfig(
                compiled_model_name=cls._BACKBONE_MODEL_NAME,
                input_info=backbone_input_info,
            ),
            RBLNCompileConfig(
                compiled_model_name=cls._HEAD_MODEL_NAME,
                input_info=head_input_info,
            ),
        ]
        if not rbln_config.use_ray_pose and model.model.cam_dec is not None:
            cam_dec_dim = model.model.cam_dec.backbone[0].in_features
            compile_cfgs.append(
                RBLNCompileConfig(
                    compiled_model_name=cls._CAM_DEC_MODEL_NAME,
                    input_info=[
                        ("feat", [batch_size, num_images, cam_dec_dim], "float32"),
                    ],
                )
            )

        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _compile_one(
        cls,
        wrapper: nn.Module,
        name: str,
        rbln_config: RBLNDepthAnything3Config,
        compile_cfg_map: dict[str, RBLNCompileConfig],
    ) -> rebel.RBLNCompiledModel:
        """Compile one wrapper and return RBLNCompiledModel."""
        return cls.compile(
            wrapper,
            rbln_compile_config=compile_cfg_map[name],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device,
        )

    @classmethod
    def get_compiled_model(
        cls, model: "DepthAnything3", rbln_config: RBLNDepthAnything3Config
    ) -> dict[str, rebel.RBLNCompiledModel]:
        """
        Compile backbone, head, and optional camera decoder to RBLN runtimes.

        Args:
            model ([`DepthAnything3`]): PyTorch model from `get_pytorch_model`.
            rbln_config ([`RBLNDepthAnything3Config`]): Config with `compile_cfgs` set (e.g. after `_update_rbln_config`).

        Returns:
            `dict[str, RBLNCompiledModel]`: Map of `compiled_model_name` (e.g. `"dinov2"`, `"head"`, `"cam_dec"`) to compiled model.
        """
        compiled_models: dict[str, rebel.RBLNCompiledModel] = {}
        compile_cfg_map = cls._build_compile_cfg_map(rbln_config.compile_cfgs)

        backbone_wrapper = cls._wrap_model_if_needed(model, rbln_config)
        compiled_models[cls._BACKBONE_MODEL_NAME] = cls._compile_one(
            backbone_wrapper, cls._BACKBONE_MODEL_NAME, rbln_config, compile_cfg_map
        )

        head_wrapper = DualDPTHeadWrapper(model.model.head)
        head_wrapper.set_image_size(rbln_config.image_size)
        compiled_models[cls._HEAD_MODEL_NAME] = cls._compile_one(
            head_wrapper, cls._HEAD_MODEL_NAME, rbln_config, compile_cfg_map
        )

        if not rbln_config.use_ray_pose and model.model.cam_dec is not None:
            cam_dec_wrapper = CameraDecWrapper(model.model.cam_dec)
            compiled_models[cls._CAM_DEC_MODEL_NAME] = cls._compile_one(
                cam_dec_wrapper, cls._CAM_DEC_MODEL_NAME, rbln_config, compile_cfg_map
            )

        return compiled_models

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNDepthAnything3Config,
    ) -> List[rebel.Runtime]:
        """Build Runtime list from compiled_models and device_map (order from compile_cfgs)."""
        expected_model_names = [
            compile_cfg.compiled_model_name for compile_cfg in rbln_config.compile_cfgs
        ]

        if any(
            model_name not in rbln_config.device_map
            for model_name in expected_model_names
        ):
            cls._raise_missing_compiled_file_error(expected_model_names)

        ret_val = [
            rebel.Runtime(
                compiled_models[i],
                tensor_type="pt",
                device=rbln_config.device_map[model_name],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for i, model_name in enumerate(expected_model_names)
        ]
        return ret_val

    @classmethod
    def _build_patch_embed_artifacts(cls, vit: nn.Module) -> dict:
        out = {
            "patch_embed_state_dict": vit.patch_embed.state_dict(),
            "patch_embed_config": {
                "img_size": vit.patch_embed.img_size,
                "patch_size": vit.patch_embed.patch_size,
                "in_chans": vit.patch_embed.in_chans,
                "embed_dim": vit.patch_embed.embed_dim,
            },
            "cls_token": vit.cls_token.data.cpu(),
            "pos_embed": vit.pos_embed.data.cpu(),
        }
        if vit.register_tokens is not None:
            out["register_tokens"] = vit.register_tokens.data.cpu()
        if getattr(vit, "camera_token", None) is not None:
            out["camera_token"] = vit.camera_token.data.cpu()
        return out

    @classmethod
    def _build_backbone_artifacts(cls, backbone: nn.Module, vit: nn.Module) -> dict:
        out = {
            "backbone_config": {
                "patch_size": vit.patch_size,
                "embed_dim": vit.embed_dim,
                "num_heads": vit.num_heads,
                "num_register_tokens": vit.num_register_tokens,
                "interpolate_antialias": vit.interpolate_antialias,
                "interpolate_offset": vit.interpolate_offset,
                "alt_start": backbone.alt_start,
                "rope_start": backbone.rope_start,
                "out_layers": backbone.out_layers,
                "cat_token": backbone.cat_token,
            },
        }
        if vit.rope is not None:
            out["rope_config"] = {"base_frequency": vit.rope.base_frequency}
        return out

    @classmethod
    def _build_head_artifacts(cls, head: nn.Module) -> dict:
        return {
            "head_config": {
                "dim_in": head.norm.normalized_shape[0],
                "patch_size": head.patch_size,
                "down_ratio": head.down_ratio,
                "activation": head.activation,
                "conf_activation": head.conf_activation,
                "pos_embed": head.pos_embed,
            },
        }

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "DepthAnything3",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNDepthAnything3Config,
    ) -> None:
        """
        Serialize patch_embed, backbone, head, and meta to torch_artifacts.pth for RBLN load.

        Args:
            model ([`DepthAnything3`]): PyTorch model to extract weights/config from.
            save_dir_path (`os.PathLike`): Directory to write artifacts into.
            subfolder (`str`): Subfolder name (e.g. `""` or model variant).
            rbln_config ([`RBLNDepthAnything3Config`]): Config with `image_size` set; used for `static_image_size` in artifacts.
        """
        backbone = model.model.backbone
        vit = backbone.pretrained
        save_dict = {
            **cls._build_patch_embed_artifacts(vit),
            **cls._build_backbone_artifacts(backbone, vit),
            **cls._build_head_artifacts(model.model.head),
        }
        rbln_config.image_size = _normalize_image_size(rbln_config.image_size)
        save_dict["static_image_size"] = tuple(rbln_config.image_size)
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def _load_patch_embed_and_token_builder(self, artifacts: dict) -> None:
        from depth_anything_3.model.dinov2.layers.patch_embed import PatchEmbed

        pe_config = artifacts["patch_embed_config"]
        patch_embed = PatchEmbed(
            img_size=pe_config["img_size"],
            patch_size=pe_config["patch_size"],
            in_chans=pe_config["in_chans"],
            embed_dim=pe_config["embed_dim"],
        )
        patch_embed.load_state_dict(artifacts["patch_embed_state_dict"])
        patch_embed.eval()
        backbone_config = artifacts["backbone_config"]
        self.patch_embed = patch_embed
        self.patch_embed_patch_size = backbone_config["patch_size"]
        self.patch_embed_interpolate_antialias = backbone_config[
            "interpolate_antialias"
        ]
        self.patch_embed_interpolate_offset = backbone_config["interpolate_offset"]
        self.patch_embed_cls_token = artifacts["cls_token"]
        self.patch_embed_pos_embed = artifacts["pos_embed"]
        self.patch_embed_register_tokens = artifacts.get("register_tokens", None)
        self.patch_embed_embed_dim = int(self.patch_embed_cls_token.shape[-1])
        self.token_builder = _TokenBuilder(
            patch_embed=self.patch_embed,
            pos_embed=self.patch_embed_pos_embed,
            cls_token=self.patch_embed_cls_token,
            register_tokens=self.patch_embed_register_tokens,
            patch_size=self.patch_embed_patch_size,
            interpolate_antialias=self.patch_embed_interpolate_antialias,
            interpolate_offset=self.patch_embed_interpolate_offset,
            embed_dim=self.patch_embed_embed_dim,
        )

    def _load_rope_and_config(self, artifacts: dict) -> None:
        backbone_config = artifacts["backbone_config"]
        head_dim = backbone_config["embed_dim"] // backbone_config["num_heads"]
        rope_config = artifacts.get("rope_config", {})
        self.rope_precomputer = RoPEPrecomputer(
            head_dim=head_dim,
            patch_size=backbone_config["patch_size"],
            num_register_tokens=backbone_config["num_register_tokens"],
            base_frequency=rope_config.get("base_frequency", 100.0),
            patch_start_idx=1,
        )
        self.backbone_config = backbone_config
        self.head_config = artifacts["head_config"]
        self.model_name = artifacts.get("model_name", "")
        self.static_image_size = artifacts.get("static_image_size", (504, 504))

    def __post_init__(self, **kwargs) -> None:
        """Load torch_artifacts, build patch_embed/rope/runtimes, and wire input/output processors."""
        artifacts = torch.load(
            self.model_save_dir / self.subfolder / "torch_artifacts.pth",
            weights_only=False,
            map_location="cpu",
        )
        self._load_patch_embed_and_token_builder(artifacts)
        self._load_rope_and_config(artifacts)
        self.dinov2_runtime = None
        self.head_runtime = None
        self.cam_dec_runtime = None
        self._assign_runtimes()
        from depth_anything_3.utils.io.input_processor import InputProcessor
        from depth_anything_3.utils.io.output_processor import OutputProcessor

        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()
        return super().__post_init__(**kwargs)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        use_ray_pose: bool = False,
        **kwargs,
    ) -> AttrDict:
        """
        Forward pass for the RBLN-optimized Depth Anything 3 model for depth estimation.

        Args:
            x (`torch.Tensor` of shape `(batch_size, num_images, channels, height, width)`):
                Input images.
            extrinsics (`torch.Tensor`, *optional*): Camera extrinsics; if provided, a warning is logged (learned tokens used).
            intrinsics (`torch.Tensor`, *optional*): Camera intrinsics; if provided, a warning is logged (learned tokens used).
            use_ray_pose (`bool`, *optional*, defaults to `False`): If `True`, pose from ray only (CPU); if `False`, pose from cam_dec only (NPU).
            **kwargs: Ignored; for API compatibility.

        Returns:
            `AttrDict`: Keys `depth`, `depth_conf`, and optionally `extrinsics`, `intrinsics` (after depad).
        """
        H_static, W_static = self.static_image_size
        images, original_hw = pad_to_static_size(x, H_static, W_static)
        B, S, _, H_pad, W_pad = images.shape
        H_orig, W_orig = original_hw
        patch_size = RBLNDepthAnything3Config.PATCH_SIZE
        ph_static = H_static // patch_size
        pw_static = W_static // patch_size
        num_special = 1 + self.backbone_config["num_register_tokens"]
        tokens, key_valid_local, key_valid_global, ph_orig, pw_orig = (
            self._build_tokens_and_key_valid(
                images,
                original_hw=(H_orig, W_orig),
                padded_hw=(H_pad, W_pad),
                num_special=num_special,
                ph_static=ph_static,
                pw_static=pw_static,
            )
        )

        rope_cos_local, rope_sin_local, rope_cos_global, rope_sin_global = (
            self.rope_precomputer.compute(
                B, S, H_static, W_static, tokens.device, tokens.dtype
            )
        )

        self._warn_unsupported_camera_inputs(
            extrinsics=extrinsics, intrinsics=intrinsics
        )

        feat0, feat1, feat2, feat3, cam_token_out = self._run_backbone(
            tokens,
            rope_cos_local,
            rope_sin_local,
            rope_cos_global,
            rope_sin_global,
            key_valid_local,
            key_valid_global,
        )

        feat0, feat1, feat2, feat3 = _replicate_edge_feats(
            feat0, feat1, feat2, feat3, ph_orig, pw_orig, ph_static, pw_static
        )

        depth, depth_conf, ray, ray_conf = self.head_runtime(feat0, feat1, feat2, feat3)

        output = AttrDict()
        output.depth = depth
        output.depth_conf = depth_conf
        self._populate_pose_outputs(
            output=output,
            ray=ray,
            ray_conf=ray_conf,
            cam_token_out=cam_token_out,
            original_hw=original_hw,
            use_ray_pose=use_ray_pose,
        )

        output = depad_output(
            output,
            original_hw,
            down_ratio=self.head_config["down_ratio"],
            patch_size=patch_size,
        )

        return output

    def inference(
        self,
        image: List,
        extrinsics: Optional[Any] = None,
        intrinsics: Optional[Any] = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        use_ray_pose: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Preprocess images, run forward, and postprocess into a single prediction object for depth estimation.

        Args:
            image: Either a list of images (single sample) or a list of such lists (batch).
                For batch execution, inputs are aligned to the compiled (batch_size, num_images).
            extrinsics: Optional camera extrinsics.
            intrinsics: Optional camera intrinsics.
            process_res: Target resolution for processing.
            process_res_method: Resize method.
            use_ray_pose: If True, pose from ray only; if False, pose from cam_dec only.
            **kwargs: Ignored; for API compatibility.

        Returns:
            Prediction object with `depth`, `depth_conf`, `extrinsics`, `intrinsics`, `processed_images`.
        """
        compiled_batch_size = None
        compiled_num_images = None
        for cfg in self.rbln_config.compile_cfgs:
            if getattr(cfg, "compiled_model_name", None) != self._BACKBONE_MODEL_NAME:
                continue
            for item in getattr(cfg, "input_info", []):
                if len(item) >= 2 and item[0] == "x":
                    shape = item[1]
                    compiled_batch_size = int(shape[0])
                    compiled_num_images = int(shape[1])
                    break
            if compiled_batch_size is not None:
                break
        if compiled_batch_size is None or compiled_num_images is None:
            raise RuntimeError(
                "Failed to infer (batch_size, num_images) from compiled model."
            )

        if (
            compiled_batch_size == 1
            and isinstance(image, (list, tuple))
            and (len(image) == 0 or not isinstance(image[0], (list, tuple)))
        ):
            image = [image]

        if not (
            isinstance(image, (list, tuple))
            and len(image) == compiled_batch_size
            and isinstance(image[0], (list, tuple))
            and all(
                isinstance(s, (list, tuple)) and len(s) == compiled_num_images
                for s in image
            )
        ):
            raise ValueError(
                "Input must be batched as List[List[image]] with compiled "
                f"(batch_size={compiled_batch_size}, num_images={compiled_num_images})."
            )
        input_samples = list(image)

        if extrinsics is not None and len(extrinsics) != compiled_batch_size:
            raise ValueError(
                "Batched inference extrinsics mismatch: expected length "
                f"{compiled_batch_size} (or None)."
            )
        if intrinsics is not None and len(intrinsics) != compiled_batch_size:
            raise ValueError(
                "Batched inference intrinsics mismatch: expected length "
                f"{compiled_batch_size} (or None)."
            )
        extrinsics_per_sample = (
            extrinsics if extrinsics is not None else [None] * compiled_batch_size
        )
        intrinsics_per_sample = (
            intrinsics if intrinsics is not None else [None] * compiled_batch_size
        )

        processed_images_per_sample = []
        processed_extrinsics_per_sample = []
        processed_intrinsics_per_sample = []
        for sample, sample_extrinsics, sample_intrinsics in zip(
            input_samples, extrinsics_per_sample, intrinsics_per_sample
        ):
            imgs_cpu_i, extrinsics_i, intrinsics_i = self.input_processor(
                sample,
                sample_extrinsics,
                sample_intrinsics,
                process_res,
                process_res_method,
            )
            processed_images_per_sample.append(imgs_cpu_i)
            processed_extrinsics_per_sample.append(extrinsics_i)
            processed_intrinsics_per_sample.append(intrinsics_i)

        imgs_cpu = torch.stack(processed_images_per_sample, dim=0)
        imgs = imgs_cpu.float()
        extrinsics_batch = (
            None
            if processed_extrinsics_per_sample[0] is None
            else torch.stack(
                [t.float() for t in processed_extrinsics_per_sample], dim=0
            )
        )
        intrinsics_batch = (
            None
            if processed_intrinsics_per_sample[0] is None
            else torch.stack(
                [t.float() for t in processed_intrinsics_per_sample], dim=0
            )
        )

        raw_output = self.forward(
            imgs, extrinsics_batch, intrinsics_batch, use_ray_pose=use_ray_pose
        )
        prediction = self.output_processor(raw_output)
        prediction.processed_images = self._processed_images_uint8(imgs_cpu)
        return prediction

    def _processed_images_uint8(self, imgs_cpu: torch.Tensor) -> Any:
        """Normalized tensor to uint8 (..., H, W, C) numpy for prediction.processed_images.
        Supports (B, C, H, W) -> (B, H, W, C) and (B, N, C, H, W) -> (B, N, H, W, C).
        """
        if imgs_cpu.dim() == 5:
            rgb = imgs_cpu.permute(0, 1, 3, 4, 2).cpu().numpy()
        else:
            rgb = imgs_cpu.permute(0, 2, 3, 1).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb = np.clip(rgb * std + mean, 0, 1)
        return (rgb * 255).astype(np.uint8)
