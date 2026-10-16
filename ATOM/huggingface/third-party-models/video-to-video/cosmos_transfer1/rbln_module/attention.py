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

import math
from typing import List, Optional

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

# ---------------------- Feed Forward Network -----------------------


def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """
    Copied from nvidia transformer_engine
    """
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    # interleaved
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def apply_rotary_pos_emb(
    t,
    freqs,
    start_positions=None,
    tensor_format="sbhd",
    interleaved=False,
    fused=True,
):
    """
    Copied from nvidia transformer_engine
    """
    max_seq_len = freqs[0].shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]
    cos_, sin_ = freqs
    cos_ = cos_.to(t.dtype)
    sin_ = sin_.to(t.dtype)

    if start_positions is not None:
        max_offset = torch.max(start_positions)
        assert max_offset + cur_seq_len <= max_seq_len, (
            f"Rotary Embeddings only suppported up to {max_seq_len} sequence length!"
        )

        cos_ = torch.concatenate(
            [cos_[i : i + cur_seq_len] for i in start_positions], dim=1
        )
        sin_ = torch.concatenate(
            [sin_[i : i + cur_seq_len] for i in start_positions], dim=1
        )

    assert cur_seq_len <= max_seq_len, (
        f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    )
    cos_ = cos_[:cur_seq_len]
    sin_ = sin_[:cur_seq_len]

    if tensor_format == "bshd":
        cos_ = cos_.transpose(0, 1)
        sin_ = sin_.transpose(0, 1)

    rot_dim = cos_.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    t = (t * cos_) + (_rotate_half(t, interleaved) * sin_)

    if t_pass.shape[-1] == 0:
        return t
    else:
        return torch.cat((t, t_pass), dim=-1)


class CustomAttention:
    def __init__(
        self,
        qkv_format,
        attention_dropout=0,
        attn_mask_type="no_mask",
    ):
        self.qkv_format = qkv_format
        self.attention_dropout = attention_dropout
        self.attn_mask_type = attn_mask_type
        if attn_mask_type not in ["no_mask", "causal", "arbitrary"]:
            raise ValueError(
                f"attention mask type {self.attn_mask_type} is not supported"
            )

    def __call__(
        self,
        q,
        k,
        v,
        attention_mask=None,
        core_attention_bias_type="no_bias",
        core_attention_bias=None,
    ):
        if self.qkv_format == "sbhd":
            q, k, v = (
                q.permute(1, 2, 0, 3),
                k.permute(1, 2, 0, 3),
                v.permute(1, 2, 0, 3),
            )
        elif self.qkv_format == "bshd":
            q, k, v = (
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
            )

        b, h, s, d = q.shape
        if self.attn_mask_type == "no_mask":
            is_causal = False
            attention_mask = None
        elif self.attn_mask_type == "causal":
            is_causal = True
            attention_mask = None
        elif self.attn_mask_type == "arbitrary":
            is_causal = False

        if core_attention_bias_type == "no_bias":
            pass
        elif core_attention_bias_type == "post_scale_bias":
            assert attention_mask is None, (
                "either attention_mask or core_attention_bias should be provided"
            )
            attention_mask = core_attention_bias
            attention_mask = attention_mask.to(dtype=q.dtype)
        elif core_attention_bias_type == "pre_scale_bias":
            assert attention_mask is None, (
                "either attention_mask or core_attention_bias should be provided"
            )
            scale_factor = 1 / math.sqrt(q.size(-1))
            attention_mask = core_attention_bias * scale_factor
            attention_mask = attention_mask.to(dtype=q.dtype)
        else:
            raise ValueError(
                f"core_attention_bias_type {core_attention_bias_type} is not supported"
            )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout,
            is_causal=is_causal,
        )

        if self.qkv_format == "sbhd":
            out = out.permute(2, 0, 1, 3)
            out = out.contiguous().view(s, b, h * d)
        elif self.qkv_format == "bshd":
            out = out.permute(0, 2, 1, 3)
            out = out.contiguous().view(b, s, h * d)

        return out


class FeedForward(nn.Module):
    """
    Transformer FFN with optional gating

    Parameters:
        d_model (int): Dimensionality of input features.
        d_ff (int): Dimensionality of the hidden layer.
        dropout (float, optional): Dropout rate applied after the activation function.
            Defaults to 0.1.
        activation (callable, optional): The activation function applied after the first
            linear layer. Defaults to nn.ReLU().
        is_gated (bool, optional): If set to True, incorporates gating mechanism to the
            feed-forward layer. Defaults to False.
        bias (bool, optional): If set to True, adds a bias to the linear layers. Defaults to True.

    Example:
        >>> ff = FeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(64, 10, 512)  # Example input tensor
        >>> output = ff(x)
        >>> print(output.shape)  # Expected shape: (64, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_gate = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_gate(x)
        else:
            x = g
        assert self.dropout.p == 0.0, "we skip dropout"
        return self.layer2(x)


class GPT2FeedForward(FeedForward):
    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = False
    ):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=nn.GELU(),
            is_gated=False,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        assert self.dropout.p == 0.0, "we skip dropout"

        x = self.layer1(x)

        x = self.activation(x)
        x = self.layer2(x)
        return x


# ---------------------- Normalization Layer -----------------------


def normalize(
    x: torch.Tensor, dim: Optional[List[int]] = None, eps: float = 0
) -> torch.Tensor:
    """
    Normalizes the input tensor along specified dimensions such that the average square
    norm of elements is adjusted.

    Args:
        x (torch.Tensor): The input tensor to normalize.
        dim (list, optional): The dimensions over which to normalize. If None,
            normalizes over all dimensions except the first.
        eps (float, optional): A small constant to ensure numerical stability during division.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Copied from huggingface transformers
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def get_normalization(name: str, channels: int):
    if name == "I":
        return nn.Identity()
    elif name == "R":
        return RMSNorm(channels, eps=1e-6)
    else:
        raise ValueError(f"Normalization {name} not found")


class BaseAttentionOp(nn.Module):
    def __init__(self):
        super().__init__()


class RegionalAttentionOp(BaseAttentionOp):
    def __init__(
        self,
        heads,
        dim_head,
        num_gqa_groups=None,
        attention_dropout=0,
        qkv_format="bshd",
        attn_mask_type="no_mask",
        tp_size=1,
        tp_group=None,
        sequence_parallel=False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.qkv_format = qkv_format
        self.tp_size = tp_size
        self.scale = dim_head**-0.5
        self.attention_dropout = attention_dropout
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group

        self.dot_product_attention = CustomAttention(
            attention_dropout=attention_dropout,
            qkv_format=qkv_format,
            attn_mask_type=attn_mask_type,
        )

    def forward(
        self,
        q,
        k,
        v,
        regional_k=None,
        regional_v=None,
        region_masks=None,
        core_attention_bias_type="no_bias",
        core_attention_bias=None,
        base_ratio=0.5,
    ):
        # Early return for non-regional case
        if regional_k is None or regional_v is None or region_masks is None:
            return self.dot_product_attention(
                q,
                k,
                v,
                attention_mask=None,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias,
            )
        # Get dimensions
        is_bshd = self.qkv_format == "bshd"
        if is_bshd:
            batch_size, seq_len, num_heads, head_dim = q.shape
        else:
            seq_len, batch_size, num_heads, head_dim = q.shape

        # Process region masks
        processed_masks = []
        prompt_len = k.shape[1] if is_bshd else k.shape[0]
        num_regions = len(regional_k)

        def preprocess_mask(mask: Tensor) -> Tensor:
            mask = mask.permute(3, 0, 1, 2)
            B, T, H, W = mask.shape
            mask = mask.unsqueeze(
                1
            )  # dummy unsqueeze since trilinear interpolation expects 5D

            mask_i = [
                torch.nn.functional.interpolate(
                    mask[:, :, :1, :, :],
                    size=(1, H // 2, W // 2),
                    mode="trilinear",
                    align_corners=False,
                )
            ]
            for wi in range(1, T, 8):
                mask_i += [
                    torch.nn.functional.interpolate(
                        mask[:, :, wi : wi + 8, :, :],
                        size=(1, H // 2, W // 2),
                        mode="trilinear",
                        align_corners=False,
                    )
                ]
            assert len(mask_i) == 16
            mask = torch.cat(mask_i, dim=2)
            mask = mask.squeeze(1)
            return (mask > 0.5).float()

        for i in range(num_regions):
            mask = region_masks[i]
            mask = mask.to(q.device)
            if mask.shape[0] != seq_len:
                mask = preprocess_mask(mask)
                mask = rearrange(mask, "b t h w ->  b (t h w)")
            processed_masks.append(mask)

        hidden_seq_len = seq_len
        regional_attention_mask = torch.zeros(
            (batch_size, hidden_seq_len, (num_regions + 1) * prompt_len),
            device=q.device,
            dtype=torch.bool,
        )
        for i, mask in enumerate(processed_masks):
            regional_attention_mask[
                :, :, (i + 1) * prompt_len : (i + 2) * prompt_len
            ] = mask.unsqueeze(-1).bool()

        regional_masks_tensor = torch.stack(processed_masks, dim=-1).bool()  # [B, S, R]
        global_mask = (
            (regional_masks_tensor.sum(dim=-1) == 0).unsqueeze(-1).bool()
        )  # [B, S, 1]
        regional_attention_mask[:, :, :prompt_len] = global_mask
        combined_k = torch.cat([k] + regional_k, dim=0)
        combined_v = torch.cat([v] + regional_v, dim=0)

        attn_bias = torch.zeros_like(regional_attention_mask, dtype=torch.float32)
        attn_bias = attn_bias.masked_fill(~regional_attention_mask, float("-inf"))
        attn_bias = attn_bias.unsqueeze(1)
        output = self.dot_product_attention(
            q,
            combined_k,
            combined_v,
            attention_mask=None,
            core_attention_bias_type="post_scale_bias",
            core_attention_bias=attn_bias,
        )

        base_output = self.dot_product_attention(
            q,
            k,
            v,
            attention_mask=None,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias=core_attention_bias,
        )
        output = output * (1 - base_ratio) + base_output * base_ratio

        if self.tp_size > 1 and not self.sequence_parallel:
            torch.distributed.all_reduce(output, group=self.tp_group)

        return output


class Attention(nn.Module):
    """
    Generalized attention impl.

    Allowing for both self-attention and cross-attention configurations depending on
    whether a `context_dim` is provided.
    If `context_dim` is None, self-attention is assumed.

    Parameters:
        query_dim (int): Dimension of each query vector.
        context_dim (int, optional): Dimension of each context vector. If None,
            self-attention is assumed.
        heads (int, optional): Number of attention heads. Defaults to 8.
        dim_head (int, optional): Dimension of each head. Defaults to 64.
        dropout (float, optional): Dropout rate applied to the output of the attention
            block. Defaults to 0.0.
        attn_op (BaseAttentionOp, optional): Custom attention operation to be used
            instead of the default.
        qkv_bias (bool, optional): If True, adds a learnable bias to query, key, and
            value projections. Defaults to False.
        out_bias (bool, optional): If True, adds a learnable bias to the output
            projection. Defaults to False.
        qkv_norm (str, optional): A string representing normalization strategies for
            query, key, and value projections. Defaults to "SSI".
        qkv_norm_mode (str, optional): A string representing normalization mode for
            query, key, and value projections. Defaults to 'per_head'. Only support
            'per_head'.

    Examples:
        >>> attn = Attention(query_dim=128, context_dim=256, heads=4, dim_head=32, dropout=0.1)
        >>> query = torch.randn(10, 128)  # Batch size of 10
        >>> context = torch.randn(10, 256)  # Batch size of 10
        >>> output = attn(query, context)  # Perform the attention operation

    Note:
        https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    """

    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attn_op: Optional[BaseAttentionOp] = None,
        qkv_bias: bool = False,
        out_bias: bool = False,
        qkv_norm: str = "SSI",
        qkv_norm_mode: str = "per_head",
        backend: str = "transformer_engine",
        qkv_format: str = "sbhd",
    ) -> None:
        super().__init__()

        self.is_selfattn = context_dim is None  # self attention

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.qkv_norm_mode = qkv_norm_mode
        self.qkv_format = qkv_format

        if self.qkv_norm_mode == "per_head":
            norm_dim = dim_head
        else:
            raise ValueError(
                f"Normalization mode {self.qkv_norm_mode} not found, only support 'per_head'"
            )

        self.backend = backend

        self.to_q = nn.Sequential(
            nn.Linear(query_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[0], norm_dim),
        )
        self.to_k = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[1], norm_dim),
        )
        self.to_v = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[2], norm_dim),
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout),
        )

        if attn_op:  # use what is given
            self.attn_op = attn_op
        elif self.backend == "transformer_engine":
            self.attn_op: BaseAttentionOp = CustomAttention(
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="no_mask",
            )
            self.regional_attn_op = RegionalAttentionOp(
                self.heads,
                self.dim_head,
                num_gqa_groups=self.heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="arbitrary",
            )
        else:
            raise ValueError(f"Backend {backend} not found")

    def cal_qkv(
        self, x, context=None, mask=None, rope_emb=None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del kwargs

        """
        self.to_q, self.to_k, self.to_v are nn.Sequential with projection + normalization layers.
        Before 07/24/2024, these modules normalize across all heads.
        After 07/24/2024, to support tensor parallelism and follow the common practice
        in the community,
        we support to normalize per head.
        To keep the checkpoint copatibility with the previous code,
        we keep the nn.Sequential but call the projection and the normalization layers separately.
        We use a flag `self.qkv_norm_mode` to control the normalization behavior.
        The default value of `self.qkv_norm_mode` is "per_head", which means we normalize per head.
        """
        if self.qkv_norm_mode == "per_head":
            q = self.to_q[0](x)
            context = x if context is None else context
            k = self.to_k[0](context)
            v = self.to_v[0](context)
            q, k, v = map(
                lambda t: rearrange(
                    t, "b ... (n c) -> b ... n c", n=self.heads, c=self.dim_head
                ),
                (q, k, v),
            )
        else:
            raise ValueError(
                f"Normalization mode {self.qkv_norm_mode} not found, only support 'per_head'"
            )

        q = self.to_q[1](q)
        k = self.to_k[1](k)
        v = self.to_v[1](v)
        if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
            q = apply_rotary_pos_emb(
                q, rope_emb, tensor_format=self.qkv_format, fused=True
            )
            k = apply_rotary_pos_emb(
                k, rope_emb, tensor_format=self.qkv_format, fused=True
            )
        return q, k, v

    def cal_attn(self, q, k, v, mask=None):
        seq_dim = self.qkv_format.index("s")
        assert q.shape[seq_dim] > 1 and k.shape[seq_dim] > 1, (
            "Seqlen must be larger than 1 for TE Attention starting with 1.8 TE version."
        )
        out = self.attn_op(
            q, k, v, core_attention_bias_type="no_bias", core_attention_bias=None
        )  # [B, Mq, H, V]
        return self.to_out(out)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rope_emb=None,
        regional_contexts=None,
        region_masks=None,
        base_ratio=0.5,
        **kwargs,
    ):
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as
                context [self attention] if None
            regional_contexts (Optional[Tensor]): Stacked regional context tensors
                [B, R, M, D] or [R, M, B, D] if THWBD format
            region_masks (Optional[Tensor]): Region masks [B, R, S] or [R, S, B] if THWBD format
        """
        q, k, v = self.cal_qkv(x, context, mask, rope_emb=rope_emb, **kwargs)

        # Early return if no regional contexts
        if regional_contexts is None or region_masks is None:
            return self.cal_attn(q, k, v, mask)

        # Process regional contexts
        regional_k = []
        regional_v = []

        # Determine format based on qkv_format
        is_bshd = self.qkv_format == "bshd"

        # Get number of regions
        num_regions = (
            regional_contexts.shape[1] if is_bshd else regional_contexts.shape[0]
        )

        # Process each region
        for i in range(num_regions):
            # Extract regional context
            reg_context = regional_contexts[:, i] if is_bshd else regional_contexts[i]

            # Ensure correct dtype
            if reg_context.dtype != context.dtype:
                reg_context = reg_context.to(dtype=context.dtype)

            _, k_regional, v_regional = self.cal_qkv(
                x, reg_context, mask, rope_emb=rope_emb, **kwargs
            )

            regional_k.append(k_regional)
            regional_v.append(v_regional)

        # Apply regional attention
        combined_attn = self.regional_attn_op(
            q,
            k,  # from global prompt
            v,  # from global prompt
            regional_k=regional_k,
            regional_v=regional_v,
            region_masks=region_masks,
            core_attention_bias_type="no_bias",
            core_attention_bias=None,
            base_ratio=base_ratio,
        )

        # Apply output projection
        return self.to_out(combined_attn)
