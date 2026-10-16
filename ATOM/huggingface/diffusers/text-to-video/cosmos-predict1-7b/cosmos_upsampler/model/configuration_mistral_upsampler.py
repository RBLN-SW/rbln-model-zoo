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

import copy
import os
from logging import Logger
from typing import Optional

import attrs
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# Common architecture specifications
BASE_CONFIG = {
    "n_kv_heads": 8,
    "norm_type": "rmsnorm",
    "norm_eps": 1e-5,
    "ffn_hidden_size": 14336,
}
COSMOS_ARCHITECTURES = {
    "1b": {
        "n_layers": 16,
        "dim": 2048,
        "n_heads": 32,
    },
    "4b": {
        "n_layers": 16,
        "dim": 4096,
        "n_heads": 32,
    },
    "12b": {
        "n_layers": 40,
        "dim": 5120,
        "n_heads": 32,
        "head_dim": 128,
    },
}

COSMOS_YARN_CONFIG = {
    "original_latent_shape": [3, 40, 64],
    "apply_yarn": True,
    "yarn_beta_fast": 4,
    "yarn_beta_slow": 1,
    "yarn_scale": 2,
}

# Llama3 architecture specifications for different model sizes
LLAMA3_ARCHITECTURES = {
    "8b": {
        "n_layers": 32,
        "dim": 4096,
        "n_heads": 32,
        "ffn_hidden_size": 14336,
    },
}
# Llama3.1 uses YaRN for long context support (context of 128k tokens)
LLAMA_YARN_CONFIG = {
    "apply_yarn": True,
    "yarn_scale": 8,
    "yarn_beta_fast": 4,
    "yarn_beta_slow": 1,
}

# Mistral architecture specifications for different model sizes
MISTRAL_ARCHITECTURES = {
    "12b": {
        "n_layers": 40,
        "dim": 5120,
        "n_heads": 32,
        "ffn_hidden_size": 14336,
        "head_dim": 128,
    },
}

PIXTRAL_VISION_ARCHITECTURES = {
    "12b": {"vision_encoder": "pixtral-12b-vit", "mm_projector": "mlp"},
}


@attrs.define(slots=True)
class ModelConfig:
    """
    A class to hold model configuration arguments.

    Args:
        dim (int): The dimensionality of the input and output of each transformer block.
        n_layers (int): Number of layers in the transformer.
        n_heads (int): Number of attention heads.
        n_kv_heads (Optional[int]): Number of key-value heads. If None, defaults to n_heads. Note: this is equivalent to
            `num_gqa_groups` in TransformerEngine, where GQA means Grouped Query Attention.
        head_dim (Optional[int]): Dimensionality of each head. If None, defaults to dim // n_heads.
        vocab_size (int): Vocabulary size.
        ffn_hidden_size (int): Hidden size for feedforward network.
        norm_eps (float): Epsilon value for normalization.
        rope_theta (float): Theta value for rotary positional embeddings.
        apply_abs_pos_emb (bool): Whether to apply absolute position embeddings.
        max_batch_size (int): Maximum batch size for inference.
        max_seq_len (int): Maximum sequence length for input text.
        fuse_qkv (bool): Whether to fuse QKV in attention. Defaults to True.
        causal_mask (bool): Whether to use causal mask. Defaults to True.
        norm_type (str): Type of normalization layer. Choices: "rmsnorm", "fused_rmsnorm", "layernorm", "np_layernorm".
        precision (str): Data type for the model.
        use_qk_normalization (bool): Whether to enable QK normalization.
        tensor_model_parallel_size (int): Tensor model parallel size. Defaults to 1.
        ckpt_dir (str): Checkpoint directory.
        ckpt_path (str): Checkpoint path.
        apply_yarn (Optional[bool]): Whether to apply YaRN (long-context extension).
        yarn_scale (Optional[float]): Scale factor for YaRN.
        yarn_beta_fast (Optional[int]): Beta fast variable for YaRN (i.e., low_freq_factor in Llama 3.1 RoPE scaling code)
        yarn_beta_slow (Optional[int]): Beta slow variable for YaRN (i.e., high_freq_factor in Llama 3.1 RoPE scaling code)
        original_seq_len (Optional[int]): Original sequence length.
        vision_encoder (Optional[str]): Vision encoder name.
        mm_projector (Optional[str]): Multi-modal projector name.
        vision_encoder_in_channels (Optional[int]): Number of channels in the input image for the vision encoder. Default is 3, you can specify to int larger than 3. E.g. if you have 4-channel images with the last channel as the alpha channel, set this to 4.
        rope_dim (Optional[str]): Dimensionality of the RoPE. Choices: "1D", "3D".
        pytorch_rope_version (Optional[str]): Version of the PyTorch RoPE implementation. Choices: "v1", "v2".
        original_latent_shape (Optional[list]): Original shape of the latent tensor needed for rope extension.
        pad_to_multiple_of (Optional[int]): Pad the position embedding to a multiple of this value.
        vision_encoder_in_channels (Optional[int]): Number of channels in the input image for the vision encoder. Default is 3.
        insert_cross_attn (bool): Whether to insert the cross-attention layers after each multi-head self-attention (MSA) layer.
        insert_cross_attn_every_k_layers (int): Insert cross-attention layers every k TransformerLayers.
        context_dim (Optional[int]): The dimensionality of cross-attention embedding, e.g., T5 embed feature dim.
        num_video_frames (Optional[int]): Number of video frames.
        video_height (Optional[int]): Raw video pixel height dimension.
        video_width (Optional[int]): Raw video pixel width dimension.
        video_latent_shape (Optional[list]): Video tokenizer output dimension, in (T,H,W).
    """

    dim: int = attrs.field(default=4096)
    n_layers: int = attrs.field(default=32)
    n_heads: int = attrs.field(default=32)
    n_kv_heads: Optional[int] = attrs.field(default=8)
    head_dim: Optional[int] = attrs.field(default=None)
    vocab_size: int = attrs.field(default=128256)
    ffn_hidden_size: int = attrs.field(default=14336)
    norm_eps: float = attrs.field(default=1e-5)
    rope_theta: float = attrs.field(default=500000)
    apply_abs_pos_emb: bool = attrs.field(default=False)
    max_batch_size: int = attrs.field(default=1)
    max_seq_len: int = attrs.field(default=8192)
    fuse_qkv: bool = attrs.field(default=False)
    causal_mask: bool = attrs.field(default=True)
    norm_type: str = attrs.field(default="rmsnorm")
    precision: str = attrs.field(default="bfloat16")
    use_qk_normalization: bool = False
    tokenizer: Optional[AutoTokenizer] = None
    tensor_model_parallel_size: int = attrs.field(default=1)
    ckpt_dir: Optional[str] = attrs.field(default=None)
    ckpt_path: Optional[str] = attrs.field(
        default=None
    )  # If not None, load the model from this path instead of ckpt_dir
    apply_yarn: Optional[bool] = attrs.field(default=False)
    yarn_scale: Optional[float] = attrs.field(default=None)
    yarn_beta_fast: Optional[int] = attrs.field(default=None)
    yarn_beta_slow: Optional[int] = attrs.field(default=None)
    original_seq_len: Optional[int] = attrs.field(default=None)
    vision_encoder: Optional[str] = attrs.field(default=None)
    vision_encoder_in_channels: Optional[int] = attrs.field(default=3)
    mm_projector: Optional[str] = attrs.field(default=None)
    rope_dim: Optional[str] = attrs.field(default="1D")
    pytorch_rope_version: Optional[str] = attrs.field(default="v2")
    original_latent_shape: Optional[list] = None
    pad_to_multiple_of: Optional[int] = None
    vision_encoder_in_channels: Optional[int] = attrs.field(default=3)
    insert_cross_attn: bool = False
    insert_cross_attn_every_k_layers: int = 1
    context_dim: Optional[int] = attrs.field(default=1024)
    # For video training
    num_video_frames: Optional[int] = None
    # Raw video pixel dimension
    video_height: Optional[int] = None
    video_width: Optional[int] = None
    # Video tokenizer output dimension, in (T,H,W), it's computed by num_video_frames/temporal_compress_factor, video_height/spatial_compression_fact, video_width/spatial_compression_fact
    video_latent_shape: Optional[list] = None

    def __getitem__(self, item):
        return getattr(self, item)


def get_model_arch_specs(
    model_size: str, model_family: str = "mistral", pretrained: bool = False
) -> dict:
    """
    Get the model architecture specifications for the given model size, model family and pretrained status.

    Args:
        model_size (str): Model size. Choices: "1b", "3b", "4b", "7b", etc.
        model_family (str): Model family. Choices: "llama", "llama3", "llama3.1", "mistral"
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        dict: A dictionary containing the model architecture specifications.
    """
    arch_specs = copy.deepcopy(BASE_CONFIG)
    model_size = model_size.lower()
    if model_family.startswith("cosmos"):
        arch_specs.update(COSMOS_ARCHITECTURES[model_size])
    elif model_family.startswith("llama"):
        arch_specs.update(LLAMA3_ARCHITECTURES[model_size])
    elif model_family in ["mistral", "pixtral"]:
        arch_specs.update(MISTRAL_ARCHITECTURES[model_size])
        if model_family == "pixtral":
            arch_specs.update(PIXTRAL_VISION_ARCHITECTURES[model_size])
    else:
        raise ValueError(f"Model family {model_family} is not supported.")

    if pretrained:
        if model_family == "cosmos":
            if model_size == "12b":
                arch_specs.update(COSMOS_YARN_CONFIG)
                Logger.debug(
                    f"Using YaRN for RoPE extension with config: {COSMOS_YARN_CONFIG}"
                )
            else:
                pass
        elif model_family in ["llama", "llama3"]:
            pretrained_specs = {
                "rope_theta": 500000,
                "max_seq_len": 8192,
                "vocab_size": 128256,
            }
            arch_specs.update(pretrained_specs)
        elif model_family == "llama3.1":
            pretrained_specs = {
                "rope_theta": 500000,
                "max_seq_len": 131072,
                "original_seq_len": 8192,
                "vocab_size": 128256,
                **LLAMA_YARN_CONFIG,
            }
            arch_specs.update(pretrained_specs)
        elif model_family == "mistral":
            assert model_size == "12b", "We only support Mistral-Nemo-12B model."
            pretrained_specs = {
                "rope_theta": 1000000,
                "max_seq_len": 128000,
                "vocab_size": 131072,
            }
            arch_specs.update(pretrained_specs)
        elif model_family == "pixtral":
            assert model_size == "12b", "We only support Pixtral 12B model."
            pretrained_specs = {
                "rope_theta": 1000000000,
                "max_seq_len": 128000,
                "vocab_size": 131072,
            }
            arch_specs.update(pretrained_specs)
        else:
            raise ValueError(
                f"Model family {model_family} doesn't have a pretrained config."
            )

    return arch_specs


def create_text_model_config(
    ckpt_dir: str,
    model_ckpt_path: str,
    tensor_model_parallel_size: int = 1,
    model_family: str = "mistral",
    model_size: str = "12b",
    precision: str = "bfloat16",
    max_seq_len: int = None,
    max_batch_size: int = 1,
    rope_dim: str = "1D",
    pytorch_rope_version: str = None,
) -> dict:
    """Create a text model for training or inference.
    Args:
        ckpt_dir (str): Checkpoint directory.
        model_ckpt_path (str): Path to the model checkpoint.
        tensor_model_parallel_size (int): Number of tensor model parallel groups.
        model_family (str): Model family. Choices: "llama", "llama3", "llama3.1", "mistral".
        model_size (str): Model size. Choices: "1b", "3b", "4b", "7b", "8b", "72b", etc.
        precision (str): Data type for the model.
        max_seq_len (int): Maximum sequence length.
        max_batch_size (int): Maximum batch size.
        rope_dim (str): RoPE dimension. Choices: "1D", "3D".
        pytorch_rope_version (str): Version of the PyTorch RoPE implementation. Choices: "v1", "v2".
    Returns:
        dict: A dictionary containing the model configuration, which can be used to instantiate the model object.
    """
    # Model size specific parameters
    model_arch_specs = get_model_arch_specs(
        model_family=model_family, model_size=model_size, pretrained=True
    )
    if max_seq_len is not None:
        # Override the max_seq_len if provided
        model_arch_specs["max_seq_len"] = max_seq_len
    if pytorch_rope_version is not None:
        model_arch_specs["pytorch_rope_version"] = pytorch_rope_version
    model_config = ModelConfig(
        max_batch_size=max_batch_size,
        precision=precision,
        ckpt_dir=ckpt_dir,
        ckpt_path=model_ckpt_path,
        use_qk_normalization=False,
        tensor_model_parallel_size=tensor_model_parallel_size,
        rope_dim=rope_dim,
        **model_arch_specs,
    )

    return model_config


def create_prompt_upsampler(
    model_id: str,
    batch_size: int = 1,
    max_seq_len: int = 1024,
    precision: int = "float32",
):
    checkpoint_dir = snapshot_download(repo_id=model_id)
    model_config = create_text_model_config(
        ckpt_dir=os.path.join(checkpoint_dir),
        model_ckpt_path=os.path.join(checkpoint_dir, "model.pt"),
        model_family="mistral",
        model_size="12b",
        precision=precision,
        max_batch_size=batch_size,
        rope_dim="1D",
        max_seq_len=max_seq_len,
        pytorch_rope_version="v1",
    )
    return model_config
