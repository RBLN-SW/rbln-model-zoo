# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil

import regex as re
import torch
from mistral_common.tokens.tokenizers.base import SpecialTokens
from safetensors.torch import load_file as safe_load_file
from transformers import (
    LlamaTokenizerFast,
    LlavaConfig,
    LlavaForConditionalGeneration,
    MistralConfig,
    PixtralVisionConfig,
)
from transformers.integrations.mistral import MistralConverter

# slightly revised from https://github.com/huggingface/transformers/blob/242bb2cafccec9f90479f5f688bca9d240b1031f/src/transformers/models/pixtral/convert_pixtral_weights_to_hf.py#L55  # noqa: E501
OLD_KEY_TO_NEW_KEY_MAPPING = {
    # Layer Normalization Weights
    r"vision_encoder.transformer.layers.(\d+).input_layernorm.weight": (
        r"model.vision_tower.transformer.layers.\1.attention_norm.weight"
    ),
    r"vision_encoder.transformer.layers.(\d+).ffn_norm.weight": (
        r"model.vision_tower.transformer.layers.\1.ffn_norm.weight"
    ),
    # Self Attention Projections
    r"vision_encoder.transformer.layers.(\d+).attention.wq.weight": (
        r"model.vision_tower.transformer.layers.\1.attention.q_proj.weight"
    ),
    r"vision_encoder.transformer.layers.(\d+).attention.wk.weight": (
        r"model.vision_tower.transformer.layers.\1.attention.k_proj.weight"
    ),
    r"vision_encoder.transformer.layers.(\d+).attention.wv.weight": (
        r"model.vision_tower.transformer.layers.\1.attention.v_proj.weight"
    ),
    r"vision_encoder.transformer.layers.(\d+).attention.wo.weight": (
        r"model.vision_tower.transformer.layers.\1.attention.o_proj.weight"
    ),
    # MLP Projections
    r"vision_encoder.transformer.layers.(\d+).feed_forward.w1.weight": (
        r"model.vision_tower.transformer.layers.\1.feed_forward.gate_proj.weight"
    ),
    r"vision_encoder.transformer.layers.(\d+).feed_forward.w2.weight": (
        r"model.vision_tower.transformer.layers.\1.feed_forward.down_proj.weight"
    ),
    r"vision_encoder.transformer.layers.(\d+).feed_forward.w3.weight": (
        r"model.vision_tower.transformer.layers.\1.feed_forward.up_proj.weight"
    ),
    # Additional mappings
    r"vision_encoder": r"model.vision_tower",
    r"vision_language_adapter.w_in": r"model.multi_modal_projector.linear_1",
    r"vision_language_adapter.w_out": r"model.multi_modal_projector.linear_2",
    r"layers.(\d+).attention.wq.weight": r"model.language_model.layers.\1.self_attn.q_proj.weight",
    r"layers.(\d+).attention.wk.weight": r"model.language_model.layers.\1.self_attn.k_proj.weight",
    r"layers.(\d+).attention.wv.weight": r"model.language_model.layers.\1.self_attn.v_proj.weight",
    r"layers.(\d+).attention.wo.weight": r"model.language_model.layers.\1.self_attn.o_proj.weight",
    r"layers.(\d+).feed_forward.w1.weight": r"model.language_model.layers.\1.mlp.gate_proj.weight",
    r"layers.(\d+).feed_forward.w2.weight": r"model.language_model.layers.\1.mlp.down_proj.weight",
    r"layers.(\d+).feed_forward.w3.weight": r"model.language_model.layers.\1.mlp.up_proj.weight",
    r"layers.(\d+).ffn_norm.weight": (
        r"model.language_model.layers.\1.post_attention_layernorm.weight"
    ),
    r"layers.(\d+).attention_norm.weight": r"model.language_model.layers.\1.input_layernorm.weight",
    r"tok_embeddings.weight": r"model.language_model.embed_tokens.weight",
    r"output.weight": r"lm_head.weight",
    r"norm.weight": r"model.language_model.norm.weight",
}


# https://github.com/huggingface/transformers/blob/242bb2cafccec9f90479f5f688bca9d240b1031f/src/transformers/integrations/mistral.py#L78
def convert_tekken_tokenizer(tokenizer_file: str):
    """Convert a "tekken" tokenizer to a fast Tokenizer."""
    # Tekken format -- need to use the Converter

    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    # Load directly using their lib
    mistral_tokenizer = MistralTokenizer.from_file(tokenizer_file)

    # Extract vocab and special tokens
    vocab = mistral_tokenizer.instruct_tokenizer.tokenizer._tekken_token2id_nospecial
    all_special = [
        token.value if hasattr(token, "value") else token
        for token in mistral_tokenizer.instruct_tokenizer.tokenizer._all_special_tokens
    ]
    specials_tokens = {
        (
            special["token_str"].value.encode("latin-1")
            if isinstance(special["token_str"], SpecialTokens)
            else special["token_str"].encode("latin-1")
        ): special["rank"]
        for special in all_special
    }
    specials_tokens.update(vocab)
    vocab = specials_tokens

    # Post-process
    all_special_token_strs = [
        special["token_str"].value
        if isinstance(special["token_str"], SpecialTokens)
        else special["token_str"]
        for special in all_special
    ]

    # Convert
    tokenizer = LlamaTokenizerFast(
        tokenizer_object=MistralConverter(
            vocab=vocab, additional_special_tokens=all_special_token_strs
        ).converted(),
    )

    tokenizer.add_special_tokens({"additional_special_tokens": all_special_token_strs})
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 11

    return tokenizer


# https://github.com/huggingface/transformers/blob/242bb2cafccec9f90479f5f688bca9d240b1031f/src/transformers/models/pixtral/convert_pixtral_weights_to_hf.py#L101
def permute_for_rope(value, n_heads, config):
    dim1 = value.shape[0]
    dim2 = config.hidden_size
    return (
        value.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


# https://github.com/huggingface/transformers/blob/242bb2cafccec9f90479f5f688bca9d240b1031f/src/transformers/models/pixtral/convert_pixtral_weights_to_hf.py#L107
def convert_dictionary(original_state_dict, vision_config, text_config):
    new_dict = {}

    all_keys = "\n" + "\n".join(original_state_dict.keys())
    old_keys = all_keys
    for old, new in OLD_KEY_TO_NEW_KEY_MAPPING.items():
        all_keys = re.sub(r"\n" + old, r"\n" + new, all_keys)

    OLD_TO_NEW = dict(zip(old_keys.split("\n"), all_keys.split("\n")))

    for key, value in original_state_dict.items():
        new_key = OLD_TO_NEW[key]
        if "vision_encoder" in key:
            _config = vision_config
            num_attention_heads = _config.num_attention_heads
        else:
            _config = text_config
            if "q_proj" in new_key:
                num_attention_heads = _config.num_attention_heads
            if "k_proj" in new_key:
                num_attention_heads = _config.num_key_value_heads

        if "q_proj" in new_key or "k_proj" in new_key:
            value = permute_for_rope(value, num_attention_heads, _config)

        new_dict[new_key] = value
    return new_dict


# https://github.com/huggingface/transformers/blob/242bb2cafccec9f90479f5f688bca9d240b1031f/src/transformers/models/pixtral/convert_pixtral_weights_to_hf.py#L136
MISTRAL_CONFIG_MAPPING = {
    "dim": "hidden_size",
    "hidden_dim": "intermediate_size",
    "n_kv_heads": "num_key_value_heads",
    "n_heads": "num_attention_heads",
    "n_layers": "num_hidden_layers",
}


# https://github.com/huggingface/transformers/blob/242bb2cafccec9f90479f5f688bca9d240b1031f/src/transformers/models/pixtral/convert_pixtral_weights_to_hf.py#L145
def convert_mistral_model(input_dir, output_dir):
    vision_config = {}
    if os.path.isfile(f"{input_dir}/params.json"):
        with open(f"{input_dir}/params.json") as f:
            param_json = json.load(f)
        vision_config = param_json.pop("vision_encoder")
        for k, v in MISTRAL_CONFIG_MAPPING.items():
            value = param_json.pop(k)
            param_json[v] = value
        if "hidden_act" not in vision_config:
            vision_config["hidden_act"] = "silu"
        text_config = MistralConfig(
            **param_json,
            hidden_act="silu",
            sliding_window=None,
            tie_word_embeddings=False,
            rms_norm_eps=param_json["norm_eps"],
            max_position_embeddings=32768,
        )
    else:
        text_config = MistralConfig(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            head_dim=128,
            hidden_act="silu",
            hidden_size=5120,
            initializer_range=0.02,
            intermediate_size=14336,
            max_position_embeddings=1024000,
            model_type="mistral",
            num_attention_heads=32,
            num_hidden_layers=40,
            num_key_value_heads=8,
            rms_norm_eps=1e-05,
            rope_theta=1000000000.0,
            sliding_window=None,
            tie_word_embeddings=False,
            vocab_size=131072,
        )
    adapter_bias = vision_config.pop("adapter_bias", True)
    vision_config = PixtralVisionConfig(**vision_config)
    config = LlavaConfig(
        vision_config,
        text_config,
        vision_feature_layer=-1,
        vision_feature_select_strategy="full",
        image_seq_length=1,
        multimodal_projector_bias=adapter_bias,
        image_token_index=vision_config.image_token_id,
    )
    config.architectures = ["LlavaForConditionalGeneration"]
    config.save_pretrained(output_dir)
    full_original_state_dict = {}
    safetensors_files = sorted(
        [file for file in os.listdir(input_dir) if file.endswith(".safetensors")]
    )
    if len(safetensors_files) == 1:
        full_original_state_dict = safe_load_file(
            f"{input_dir}/consolidated.safetensors"
        )
    else:
        for file in safetensors_files:
            loaded_dict = safe_load_file(f"{input_dir}/{file}")
            full_original_state_dict.update(loaded_dict)

    new_dict = convert_dictionary(full_original_state_dict, vision_config, text_config)
    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config)
    model.load_state_dict(new_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)

    # copy tekken.json
    if os.path.isfile(f"{input_dir}/tekken.json"):
        shutil.copy(f"{input_dir}/tekken.json", f"{output_dir}/tekken.json")
