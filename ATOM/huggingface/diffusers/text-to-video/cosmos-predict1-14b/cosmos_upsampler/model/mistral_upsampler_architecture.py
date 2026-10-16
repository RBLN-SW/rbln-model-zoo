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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_mistral_upsampler import ModelConfig
from .modeling_util import create_norm


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
    mask: Optional[torch.Tensor] = None,
    is_causal: Optional[bool] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    PyTorch's native implementation of Flash Attention 2.

    If `is_causal` is given, then the causal attention mask is applied accordingly:
    - If `is_causal` is True, the standard upper-left causal attention masking is applied.
    - If `is_causal` is False, no attention mask is applied, unless an explicit mask tensor is
      provided (i.e., `mask is not None`).

    If `is_causal` is not given (i.e., `is_causal is None`), then the attention mask is applied
    based on the provided mask tensor:
    - If no explicit attention mask is given (i.e., `mask is None`), `is_causal` is set to True,
    leading to the standard upper-left causal attention masking.
    - If an attention mask is given (i.e., `mask is not None`), the provided mask is used,
    and `is_causal` is set to False.

    Args:
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        v (torch.Tensor): Value tensor
        head_dim (int): Dimension of each attention head
        mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
        is_causal (Optional[bool], optional): Whether to apply causal attention mask. Defaults to None.
        dropout_p (float, optional): Dropout rate. Defaults to 0.0.

    Returns:
        torch.Tensor: Output tensor after applying scaled dot-product attention
    """
    scale = 1.0 / math.sqrt(head_dim)
    if is_causal is None:
        is_causal = mask is None

    y = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=dropout_p,
        scale=scale,
        is_causal=is_causal,
    )
    return y.transpose(1, 2).contiguous()


class RotaryPositionEmbedding1DV1(nn.Module):
    """
    Rotary Position Embedding that works in the same way as
    mistral_inference (https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/rope.py)
    or llama3 (https://github.com/meta-llama/llama3/blob/main/llama/model.py)

    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mscale = 1.0
        self.config = config
        self.dim = (
            config.head_dim
            if config.head_dim
            else self.config.dim // self.config.n_heads
        )
        self.max_position_embeddings = config.max_seq_len
        self.max_seq_len_cached = self.max_position_embeddings

        self.inv_freq = 1.0 / (
            (
                getattr(self.config, "rope_theta", None)
                or self.config.rope_parameters["rope_theta"]
            )
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.seq = torch.arange(self.max_seq_len_cached, dtype=torch.float32)

        self.freqs = torch.einsum("i,j->ij", self.seq, self.inv_freq)
        emb = torch.stack((self.freqs, self.freqs), dim=-1).reshape(
            *self.freqs.shape[:-1], -1
        )
        self.register_buffer(
            "cos_cached", (emb.cos() * self.mscale)[None, :, None, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * self.mscale)[None, :, None, :], persistent=False
        )

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dimensions of the input tensor."""
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        output = torch.stack((-x2, x1), dim=-1).reshape(*x.shape)
        return output

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the rotary position embedding.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            input_pos (Optional[torch.Tensor]): Starting position for the sequence.
            seq_len (Optional[int]): Length of the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
        """
        if input_pos is not None:
            cos_cached = self.cos_cached[:, input_pos]
            sin_cached = self.sin_cached[:, input_pos]
        else:
            assert self.cos_cached.shape[1] >= seq_len, (
                f"Invalid sequence length; cos_cached.shape {self.cos_cached.shape}, seq_len {seq_len}."
            )
            cos_cached = self.cos_cached[:, :seq_len, ...]
            sin_cached = self.sin_cached[:, :seq_len, ...]

        xq = q * cos_cached + self.rotate_half(q) * sin_cached
        xk = k * cos_cached + self.rotate_half(k) * sin_cached

        return xq.type_as(q), xk.type_as(k)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        """
        Initializes the multilayer perceptron (MLP) module.

        Args:
            dim: The input and output dimensionality.
            hidden_dim: The dimensionality of the hidden layer.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLP module.

        Args:
            x: The input tensor of shape (batch_size, dim).

        Returns:
            The output tensor of shape (batch_size, dim).
        """
        output = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return output


class Attention(nn.Module):
    """
    Attenion layer with KV cache.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        context_dim: Optional[int] = None,
        attn_type: str = "self",
    ):
        """
        Initializes the GQA module.
        """
        super().__init__()
        assert attn_type in [
            "self",
            "cross",
            "full",
        ], f"Invalid attention type: {attn_type}"
        self.config = config
        self.layer_id = layer_id
        self.attn_type = attn_type
        context_dim = config.dim if context_dim is None else context_dim

        self.dim = config.dim
        self.context_dim = context_dim
        self.n_kv_heads = (
            config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        )
        self.n_local_kv_heads = self.n_kv_heads
        self.n_local_heads = config.n_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = (
            config.dim // config.n_heads if config.head_dim is None else config.head_dim
        )
        self.causal_mask = config.causal_mask
        self.fuse_qkv = config.fuse_qkv
        self.precision = config.precision

        self.wq = nn.Linear(self.dim, self.n_local_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(
            context_dim, self.n_local_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            context_dim, self.n_local_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(self.n_local_heads * self.head_dim, self.dim, bias=False)

        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len
        self.use_qk_normalization = config.use_qk_normalization

        if self.attn_type == "self":
            # Cache for key and value tensors
            self.init_kv_cache()

        # QK normalization layers
        if self.use_qk_normalization:
            self.q_norm = create_norm(
                config.norm_type, dim=self.head_dim, eps=config.norm_eps
            )
            self.k_norm = create_norm(
                config.norm_type, dim=self.head_dim, eps=config.norm_eps
            )

        self.to(dtype=getattr(torch, self.precision))

    def init_kv_cache(self, dtype=None):
        cache_shape = (
            self.max_batch_size,
            self.n_local_kv_heads,
            self.max_seq_len,
            self.head_dim,
        )
        if dtype is None:
            dtype = getattr(torch, self.precision)
        if self.attn_type == "self":
            self.cache_k = torch.zeros(cache_shape, dtype=dtype)
            self.cache_v = torch.zeros(cache_shape, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionEmbedding1DV1,
        input_pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of GQA.

        Args:
            x: The input tensor of shape (batch_size, seq_len, dim).
            rope: The rotary positional embedding module.
            input_pos: The starting position of the current sequence.
            mask: The attention mask tensor.
            context: The context tensor of shape (batch_size, context_len, dim).

        Returns:
            The output tensor after applying GQA.
        """
        bsz, seqlen, _ = x.shape

        # Use one single module to handle both self-attn and cross-attn
        context = x
        context_len = seqlen

        if self.fuse_qkv:
            q_size = self.n_local_heads * self.head_dim
            kv_size = self.n_local_kv_heads * self.head_dim
            xq, xk, xv = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
        else:
            # Compute query, key, and value projections
            xq, xk, xv = self.wq(x), self.wk(context), self.wv(context)

        # Reshape projections
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, context_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, context_len, self.n_local_kv_heads, self.head_dim)

        # QK normalization
        if self.use_qk_normalization:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # Apply rotary positional embeddings to queries and keys
        # Only apply RoPE to self-attention!
        if self.attn_type in ["self", "full"]:
            xq, xk = rope(xq, xk, input_pos, seqlen)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))
        # xq: (bs, n_local_heads, seqlen, head_dim)
        # xk: (bs, n_kv_heads, cache_len + context_len, head_dim)
        # xv: (bs, n_kv_heads, cache_len + context_len, head_dim)
        if self.attn_type == "self":
            # Update cache with current key and value tensors
            assert input_pos is not None
            self.cache_k[:bsz, :, input_pos] = xk
            self.cache_v[:bsz, :, input_pos] = xv
            keys, values = (
                self.cache_k[:bsz, :, :],
                self.cache_v[:bsz, :, :],
            )
        else:
            keys, values = xk, xv

        # Repeat keys and values if necessary
        keys = keys.repeat_interleave(
            self.n_rep, dim=1
        )  # (bs, n_local_heads, cache_len + context_len, head_dim)
        values = values.repeat_interleave(
            self.n_rep, dim=1
        )  # (bs, n_local_heads, cache_len + context_len, head_dim)

        # For self-attention, `is_causal` should be set to False when KV cache is pre-computed and used,
        # since the masking is handled outside this attention module.
        # For cross-attention, it's always full-attn without causal mask
        is_causal = False
        output = scaled_dot_product_attention(
            xq,
            keys,
            values,
            head_dim=self.head_dim,
            mask=mask,
            is_causal=is_causal,
            dropout_p=0.0,
        )
        output = output.view(bsz, seqlen, -1)
        output = self.wo(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config

        self.attention = Attention(config, self.layer_id)

        self.feed_forward = MLP(
            dim=self.config.dim,
            hidden_dim=self.config.ffn_hidden_size,
        )
        self.attention_norm = create_norm(
            self.config.norm_type, dim=self.config.dim, eps=self.config.norm_eps
        )
        self.ffn_norm = create_norm(
            self.config.norm_type, dim=self.config.dim, eps=self.config.norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionEmbedding1DV1,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the TransformerBlock module.

        Args:
            x: The input tensor.
            input_pos: The position of the current sequence. Used in inference (with KV cache) only.
            freqs_cis: The precomputed frequency values for rotary position embeddings.
            mask: The attention mask tensor.
            context (Optional[torch.Tensor]): The context tensor added via cross-attn.
            context_mask (Optional[torch.Tensor]): The context cross-attn mask tensor.

        Returns:
            The output tensor after applying the transformer block.
        """
        # Apply attention and residual connection
        h = x + self.attention(
            self.attention_norm(x), rope=rope, input_pos=input_pos, mask=mask
        )

        # Apply feed-forward network and residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class T2WTransformer(nn.Module):
    """
    The Transformer network consisting of transformer blocks.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.precision = getattr(torch, self.config.precision)

        # Token embeddings
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.dim).to(
            self.precision
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(layer_id, self.config).to(self.precision)
                for layer_id in range(self.config.n_layers)
            ]
        )

        # Final layer normalization
        self.norm = create_norm(
            self.config.norm_type, dim=self.config.dim, eps=self.config.norm_eps
        ).to(self.precision)

        # Rotary position embeddings
        self.rope = RotaryPositionEmbedding1DV1(config)

        # Causal mask
        self.causal_mask = torch.tril(
            torch.ones(
                self.config.max_seq_len, self.config.max_seq_len, dtype=torch.bool
            )
        )

        # Output projection
        self.output = nn.Linear(self.config.dim, self.config.vocab_size, bias=False).to(
            self.precision
        )

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the Transformer module.

        Args:
            tokens (torch.Tensor, optional): The input tensor of token IDs.
            input_pos (Optional[torch.Tensor]): The position of the current sequence. Used in inference with KV cache.
            token_embeddings (torch.Tensor, optional): Precomputed token embeddings. If provided, tokens should be None.
            context (Optional[torch.Tensor]): The context tensor added via cross-attn.
            context_mask (Optional[torch.Tensor]): The context cross-attn mask tensor.
        Returns:
            The output tensor after applying the transformer layers.
        """
        # Token embeddings
        assert tokens is None or token_embeddings is None, (
            "Either tokens or token_embeddings should be provided, not both."
        )

        if token_embeddings is None:
            h = self.tok_embeddings(tokens)
        else:
            h = token_embeddings

        # Create attention mask
        assert input_pos is not None, "input_pos must be provided for inference"
        mask = self.causal_mask[input_pos]
        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask[None, None, :, :]

        # Apply transformer layers
        for layer in self.layers:
            h = layer(x=h, input_pos=input_pos, mask=mask, rope=self.rope)

        # Apply final layer normalization
        h = self.norm(h)

        # Output linear projection
        output = self.output(h)

        return output
