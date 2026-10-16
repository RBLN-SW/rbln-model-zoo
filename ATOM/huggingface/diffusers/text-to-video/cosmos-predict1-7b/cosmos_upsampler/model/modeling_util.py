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

from typing import Optional

import torch
import torch.nn as nn

# Substrings to ignore when processing state dicts
substrings_to_ignore = [
    "_extra_state",  # Extra states (BytesIO type) added by TransformerEngine for FP8 handling
]


def process_state_dict(
    state_dict: dict[str, torch.Tensor],
    device: str = None,
    dtype: torch.dtype = None,
    prefix_to_remove: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """
    - Remove items with substring "_extra_state" in keys (TransformerEngine adds these for FP8)
    - Move tensors to specified device and dtype if provided



    Args:
        state_dict (dict[str, torch.Tensor]): The state dict to process
        device (str, optional): The device to move tensors to. Defaults to None.
        dtype (torch.dtype, optional): The dtype to move tensors to. Defaults to None.
        prefix_to_remove (str, optional): The prefix to remove from the keys of the state dict. Defaults to None.
    Returns:
        dict[str, torch.Tensor]: The processed state dict
    """
    new_state_dict = {}
    tensor_kwargs = {}
    if device is not None:
        tensor_kwargs["device"] = device
    if dtype is not None:
        tensor_kwargs["dtype"] = dtype

    for key, value in state_dict.items():
        # Check if any of the substrings to ignore are in the key
        skip = False
        for substr in substrings_to_ignore:
            if substr in key:
                skip = True
                break
        if skip:
            continue
        if len(tensor_kwargs) > 0:
            value = value.to(**tensor_kwargs)
        if prefix_to_remove is not None and key.startswith(prefix_to_remove):
            key = key[len(prefix_to_remove) :]
        new_state_dict[key] = value
    return new_state_dict


def sample_top_p(logits, temperature, top_p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        logits (torch.Tensor): Logits of the probability distribution.
        temperature (float): Temperature for sampling.
        top_p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
    # Sort the probabilities in descending order and get their indices.
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # Compute the cumulative sum of the sorted probabilities.
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Create a mask where the cumulative probability exceeds the threshold p.
    mask = probs_sum - probs_sort > top_p
    # Set the probabilities that exceed the threshold to 0.
    probs_sort[mask] = 0.0
    # Renormalize the remaining probabilities so they sum to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample from the renormalized probability distribution.
    # next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = multinomial_sample_one_no_sync(probs_sort, dtype=torch.int64)
    # Gather the indices of the sampled tokens.
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token


def multinomial_sample_one_no_sync(probs_sort, dtype=torch.int):
    """
    Multinomial sampling without a cuda synchronization.
    Source: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=dtype)


def logits_to_probs(
    logits,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample_top_k(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    """
    Sample from the logits using top-k sampling.
    Source: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    # logits: [batch_size, seq_len, vocab_size]
    if temperature == 0.0:
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    else:
        probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next


def create_norm(norm_type: str, dim: int, eps: float = 1e-6):
    """
    Creates the specified normalization layer based on the norm_type.
    Adopted from TorchTriton: https://github.com/pytorch/torchtitan/blob/main/torchtitan/cosmos_predict1/norms.py
    Args:
        norm_type (str): The type of normalization layer to create.
            Supported types: 1. rmsnorm 2. fused_rmsnorm 3. layernorm 4. np_layernorm
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.
    Returns:
        The created normalization layer.
    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    norm_type = norm_type.lower()  # Normalize to lowercase

    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps, compile=False)
    elif norm_type == "compiled_rmsnorm":
        return RMSNorm(dim, eps=eps, compile=True)
    elif norm_type == "fused_rmsnorm":
        raise NotImplementedError("Fused RMSNorm is not supported yet.")
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.
    Reference implementation: https://github.com/pytorch/torchtitan/blob/main/torchtitan/cosmos_predict1/norms.py
    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        compile (bool, optional): Whether to compile the forward function. Default is False.
    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6, compile: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.rmsnorm_fn = (
            torch.compile(self.compute_rmsnorm, fullgraph=True)
            if compile
            else self.compute_rmsnorm
        )

    @staticmethod
    def compute_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
        def _norm(x, eps):
            # Computes the root-mean-square norm of the input tensor.
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        output = _norm(x.float(), eps).type_as(x)
        return output * weight

    def forward(self, x: torch.Tensor):
        return self.rmsnorm_fn(x, self.weight, self.eps)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
