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

import time
from typing import List, Optional, Set, Union

import torch
import torch.nn as nn

from .configuration_mistral_upsampler import create_prompt_upsampler
from .mistral_upsampler_architecture import T2WTransformer
from .modeling_util import process_state_dict, sample_top_k, sample_top_p


class MistralNeMoForTextUpsampler(nn.Module):
    def __init__(
        self,
        model_id: str,
        batch_size: int = 1,
        max_seq_len: int = 1024,
        **kwargs,
    ):
        super().__init__()

        model_config = create_prompt_upsampler(
            model_id=model_id,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )

        # setting dtype
        dtype_map = {"auto": "float32"}
        precision = kwargs.get("torch_dtype", "auto")
        precision = dtype_map.get(precision, precision)
        model_config.precision = precision
        self.dtype = precision

        n_layers = kwargs.get("num_hidden_layers", None) or kwargs.get("n_layers", None)
        if n_layers is not None:
            model_config.n_layers = n_layers

        self.config = model_config

        self.model = T2WTransformer(model_config)

        self.load_model()

    def load_model(self):
        ckpt_path = self.config.ckpt_path
        checkpoint = torch.load(
            ckpt_path, map_location="cpu", mmap=True, weights_only=True
        )
        llm_checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint
        llm_checkpoint = process_state_dict(llm_checkpoint, prefix_to_remove="model.")
        self.model.load_state_dict(llm_checkpoint, strict=False)

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            tokens=tokens, input_pos=input_pos, token_embeddings=token_embeddings
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: Union[List[List[int]], torch.Tensor],
        max_gen_len: int,
        stop_tokens: Optional[Set[int]],
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 0.0,
        num_gen_seq: int = 1,
        echo: bool = False,
        seed: int = None,
        verbose: bool = True,
        return_logit: bool = False,
    ):
        """
        Autoregressive generation built upon the gpt-fast implementation (https://github.com/pytorch-labs/gpt-fast).

        Args:
            prompt_tokens (List[List[int]] | torch.Tensor): A single prompt of shape (1, seq_len).
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_k (int, optional): Top-k value for top-k sampling. Defaults to None.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to None.
            num_gen_seq (int, optional): Number of outputs to generate given the same prompt. Defaults to 1. When temperature == 0, num_gen_seq must be 1 because the generation is deterministic.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
            logit_clipping_range (list, optional): Range of logits to clip. Defaults to [].
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            verbose (bool, optional): Flag indicating whether to print the the time. Defaults to False.
        """
        if temperature == 0.0:
            print(
                "WARNING: top_k and top_p are automatically set to None since temperature is 0.0 and sampling (do_sample) is False, "
                + "consistent with the Hugging Face generate method behavior."
            )
            top_k, top_p = None, None

        if top_p is not None and top_k is not None:
            print(
                "WARNING: Both top_p and top_k have been specified. In accordance with the sampling policy, only top_p will be used."
            )

        if return_logit:
            final_logits = []

        if seed is not None:
            torch.random.seed(seed)

        params = self.config

        if isinstance(prompt_tokens, list):
            prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda")
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens.view(1, -1)
        else:
            assert prompt_tokens.ndim == 2, (
                f"prompt_tokens has shape {prompt_tokens.shape}"
            )

        batch_size, prompt_len = prompt_tokens.shape

        # Check max_seq_len vs max_gen_len + prompt_len
        total_len = min(params.max_seq_len, max_gen_len + prompt_len)
        if max_gen_len + prompt_len > params.max_seq_len:
            print(
                f"max_gen_len + prompt_len={max_gen_len + prompt_len} exceeds max_seq_len={params.max_seq_len}, truncate max_gen_len to {params.max_seq_len - prompt_len}"
            )
            max_gen_len = params.max_seq_len - prompt_len

        if num_gen_seq > 1:
            assert batch_size == 1, (
                f"num_gen_seq > 1 is only supported for a single prompt, got {len(prompt_tokens)} prompts"
            )
            print(f"Generating {num_gen_seq} sequences with the same prompt")
            assert num_gen_seq <= params.max_batch_size, (
                f"num_gen_seq={num_gen_seq} exceeds max_batch_size={params.max_batch_size}"
            )
            # repeat the prompt tokens for num_gen_seq times
            prompt_tokens = prompt_tokens.repeat(num_gen_seq, 1)
            assert prompt_tokens.shape == (
                num_gen_seq,
                prompt_len,
            ), (
                f"prompt_tokens must be of shape (num_gen_seq, seq_len), got {prompt_tokens.shape}"
            )
            batch_size = len(prompt_tokens)

        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(
            batch_size,
            total_len,
            dtype=prompt_tokens.dtype,
            device=prompt_tokens.device,
        )
        empty[:, :prompt_len] = prompt_tokens
        seq = empty
        input_pos = torch.arange(0, prompt_len)

        if verbose:
            prefill_start = time.time()

        prompt_token_embeddings = None

        # Prefill stage
        logits = self.model(
            input_pos=input_pos,
            tokens=prompt_tokens if prompt_token_embeddings is None else None,
            token_embeddings=prompt_token_embeddings,
        )
        if return_logit:
            final_logits.append(logits[:, -1, :])

        if top_p is not None:
            next_token = sample_top_p(logits, temperature=temperature, top_p=top_p)
        else:
            next_token = sample_top_k(logits, temperature=temperature, top_k=top_k)

        if verbose:
            prefill_time = time.time() - prefill_start

        seq[:, [prompt_len]] = next_token.to(dtype=seq.dtype)
        input_pos = torch.tensor([prompt_len], dtype=torch.long, device="cpu")
        stop_tokens = torch.tensor(list(stop_tokens), dtype=torch.long, device="cpu")

        if verbose:
            decode_start = time.time()

        # Decode stage
        generated_tokens = []
        cur_token = next_token.view(batch_size, -1)
        if stop_tokens is not None:
            # Indicator for whether the EOS token (stop token) has been reached for each sample in the batch
            eos_reached = torch.tensor([False] * batch_size, device="cpu")

        for t in range(max_gen_len - 1):
            logits = self.model(
                input_pos=input_pos,
                tokens=cur_token,
                token_embeddings=None,
            )
            input_pos += 1
            if return_logit:
                final_logits.append(logits[:, -1, :])

            if top_p is not None:
                next_token = sample_top_p(logits, temperature=temperature, top_p=top_p)
            else:
                next_token = sample_top_k(logits, temperature=temperature, top_k=top_k)

            generated_tokens.append(next_token.clone())
            if stop_tokens is not None and len(stop_tokens) > 0:
                eos_reached = eos_reached | (torch.isin(next_token, stop_tokens))
                if eos_reached.all():
                    break
            cur_token = next_token.clone()

        gen_len = len(generated_tokens)
        if verbose:
            decode_time = time.time() - decode_start
            prefill_throughput = prompt_len / prefill_time
            decode_throughput = gen_len / decode_time
            print(
                f"[Prefill] Time: {prefill_time:.2f}s; Throughput: {prefill_throughput:.2f} tokens/s"
            )
            print(
                f"[Decode] Time: {decode_time:.2f}s; Throughput: {decode_throughput:.2f} tokens/s"
            )

        generated_tokens = torch.cat(generated_tokens, dim=1)

        print(f"generated_tokens: {generated_tokens.shape}")
        seq = seq[:, : prompt_len + 1 + gen_len]
        seq[:, prompt_len + 1 :] = generated_tokens
        if not echo:
            seq = seq[:, prompt_len:]

        if return_logit:
            return seq, final_logits
        else:
            return seq
