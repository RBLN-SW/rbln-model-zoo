# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn as nn
from optimum.rbln.transformers.models.decoderonly.configuration_lora import (
    RBLNLoRAConfig,
)
from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
)

from .configuration_mistral_upsampler import RBLNMistralNeMoForTextUpsamplerConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input tensor."""
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x_reshaped[..., 0]
    x2 = x_reshaped[..., 1]
    output = torch.stack((-x2, x1), dim=-1).reshape(*x.shape)
    return output


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MistralNeMoAttention(DecoderOnlyAttention):
    def __init__(
        self,
        self_attn,
        rbln_config: "RBLNMistralNeMoForTextUpsamplerConfig",
        is_sliding: bool = False,
    ):
        nn.Module.__init__(self)
        self.rbln_config = rbln_config
        self.layer_idx = self_attn.layer_id
        self.num_key_value_heads = self_attn.n_kv_heads
        self.num_heads = self_attn.config.n_heads
        self.head_dim = self_attn.head_dim
        self.qk_norm = self_attn.use_qk_normalization
        self.scale = torch.tensor(self.get_attn_scale(self_attn))
        self._phase = "prefill"
        self.is_sliding = is_sliding
        self.attn_impl = rbln_config.attn_impl
        self.lora_config = None

        if self.is_sliding and self.attn_impl != "eager":
            raise NotImplementedError(
                "Sliding window attention is only supported with eager attention."
            )

        self.kvcache_partition_len = rbln_config.kvcache_partition_len
        self.kvcache_block_size = rbln_config.kvcache_block_size

        setattr(self, self.get_attention_name(), self.create_attention_op())
        self.__post_init__(self_attn)

    def __post_init__(self, self_attn):
        self.q_proj = self_attn.wq
        self.k_proj = self_attn.wk
        self.v_proj = self_attn.wv
        self.o_proj = self_attn.wo
        if self.qk_norm:
            self.q_norm = getattr(self_attn, "q_norm", None)
            self.k_norm = getattr(self_attn, "k_norm", None)

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)


class MistralNeMoLayer(DecoderOnlyLayer):
    def __init__(
        self,
        layer,
        self_attn: "DecoderOnlyAttention",
        lora_config: Optional[
            RBLNLoRAConfig
        ] = None,  # NOTE: This model does not support LoRA
    ):
        nn.Module.__init__(self)
        self.pre_attention_layernorm = layer.attention_norm
        self.post_attention_layernorm = layer.ffn_norm
        self.mlp = layer.feed_forward
        self.self_attn = self_attn
        self._phase = "prefill"
        self.lora_config = lora_config


class MistralNeMoModel(DecoderOnlyModel):
    _EMBEDDING_ATTRS = ["tok_embeddings"]


class MistralNeMoForTextUpsampler(DecoderOnlyForCausalLM):
    pass


class RotaryEmbedding1DV1(nn.Module):
    def __init__(
        self,
        config,
        max_seq_len_cached: int,
    ):
        super().__init__()
        self.mscale = 1.0
        self.config = config
        self.dim = (
            config.head_dim
            if config.head_dim
            else self.config.dim // self.config.n_heads
        )

        self.inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64) / self.dim)
        )
        cache_position = torch.arange(0, max_seq_len_cached, dtype=torch.float32)
        cache_position_expanded = cache_position[:, None]

        inv_freq_expanded = self.inv_freq[None, :]
        self.freqs = cache_position_expanded.float() @ inv_freq_expanded.float()

        emb = torch.stack((self.freqs, self.freqs), dim=-1).reshape(
            *self.freqs.shape[:-1], -1
        )
        self.register_buffer("_cos_cached", (emb.cos() * self.mscale), persistent=False)
        self.register_buffer("_sin_cached", (emb.sin() * self.mscale), persistent=False)

    def forward(self, x, seq_len):
        return (
            self._cos_cached[:seq_len].to(dtype=x.dtype),
            self._sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MistralNeMoForTextUpsamplerWrapper(DecoderOnlyWrapper):
    def get_attn_layer(self, layer):
        return layer.attention

    def get_rbln_attn_class(self):
        return MistralNeMoAttention

    def get_rbln_layer_class(self):
        return MistralNeMoLayer

    def get_rbln_model_class(self):
        return MistralNeMoModel

    def get_rbln_causal_lm_class(self):
        return MistralNeMoForTextUpsampler

    def get_rotary_emb(self, max_seq_len):
        return RotaryEmbedding1DV1(config=self.config, max_seq_len_cached=max_seq_len)
