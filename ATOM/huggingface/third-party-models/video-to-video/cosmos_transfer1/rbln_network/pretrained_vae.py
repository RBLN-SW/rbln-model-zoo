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

import os

import torch
from einops import rearrange


class RBLNVideoJITTokenizer:
    def __init__(self, video_vae, encoder_runtime, decoder_runtime):
        self._temporal_compress_factor
        self.spatial_compression_factor
        self.spatial_resolution
        self.latent_ch
        self.latent_chunk_duration
        self.pixel_chunk_duration
        self.max_enc_batch_size

    def register_mean_std(self, vae_dir: str) -> None:
        latent_mean, latent_std = torch.load(
            os.path.join(vae_dir, "mean_std.pt"),
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        latent_mean = latent_mean.view(self.latent_ch, -1)[
            :, : self.latent_chunk_duration
        ]
        latent_std = latent_std.view(self.latent_ch, -1)[
            :, : self.latent_chunk_duration
        ]

        target_shape = [1, self.latent_ch, self.latent_chunk_duration, 1, 1]

        self.latent_mean = latent_mean.reshape(*target_shape)
        self.latent_std = latent_std.reshape(*target_shape)

    def transform_encode_state_shape(self, state: torch.Tensor) -> torch.Tensor:
        """
        Rearranges the input state tensor to the required shape for encoding video data.
        Mainly for chunk based encoding
        """
        B, C, T, H, W = state.shape
        assert T % self.pixel_chunk_duration == 0, (
            f"Temporal dimension {T} is not divisible by chunk_length {self.pixel_chunk_duration}"
        )
        return rearrange(
            state, "b c (n t) h w -> (b n) c t h w", t=self.pixel_chunk_duration
        )

    def transform_decode_state_shape(self, latent: torch.Tensor) -> torch.Tensor:
        B, _, T, _, _ = latent.shape
        assert T % self.latent_chunk_duration == 0, (
            f"Temporal dimension {T} is not divisible by chunk_length {self.latent_chunk_duration}"
        )
        return rearrange(
            latent, "b c (n t) h w -> (b n) c t h w", t=self.latent_chunk_duration
        )

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        if self._temporal_compress_factor == 1:
            _, _, origin_T, _, _ = state.shape
            state = rearrange(state, "b c t h w -> (b t) c 1 h w")
        B, C, T, H, W = state.shape
        state = self.transform_encode_state_shape(state)
        # use max_enc_batch_size to avoid OOM
        if state.shape[0] > self.max_enc_batch_size:
            latent = []
            for i in range(0, state.shape[0], self.max_enc_batch_size):
                latent.append(super().encode(state[i : i + self.max_enc_batch_size]))
            latent = torch.cat(latent, dim=0)
        else:
            latent = super().encode(state)

        latent = rearrange(latent, "(b n) c t h w -> b c (n t) h w", b=B)
        if self._temporal_compress_factor == 1:
            latent = rearrange(latent, "(b t) c 1 h w -> b c t h w", t=origin_T)
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decodes a batch of latent representations into video frames by applying temporal chunking.
        Similar to encode, it handles video data by processing smaller temporal chunks to
        reconstruct the original video dimensions.

        It can also decode single frame image data.

        Args:
            latent (torch.Tensor): The latent space tensor containing encoded video data.

        Returns:
            torch.Tensor: The decoded video tensor reconstructed from latent space.
        """
        if self._temporal_compress_factor == 1:
            _, _, origin_T, _, _ = latent.shape
            latent = rearrange(latent, "b c t h w -> (b t) c 1 h w")
        B, _, T, _, _ = latent.shape
        latent = self.transform_decode_state_shape(latent)
        # use max_enc_batch_size to avoid OOM
        if latent.shape[0] > self.max_dec_batch_size:
            state = []
            for i in range(0, latent.shape[0], self.max_dec_batch_size):
                state.append(super().decode(latent[i : i + self.max_dec_batch_size]))
            state = torch.cat(state, dim=0)
        else:
            state = super().decode(latent)
        assert state.shape[2] == self.pixel_chunk_duration
        state = rearrange(state, "(b n) c t h w -> b c (n t) h w", b=B)
        if self._temporal_compress_factor == 1:
            return rearrange(state, "(b t) c 1 h w -> b c t h w", t=origin_T)
        return state

    @property
    def pixel_chunk_duration(self) -> int:
        return self._pixel_chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        # return self._latent_chunk_duration
        assert (
            self.pixel_chunk_duration - 1
        ) % self.temporal_compression_factor == 0, (
            f"Pixel chunk duration {self.pixel_chunk_duration} is not divisible by "
            f"latent chunk duration {self.latent_chunk_duration}"
        )
        return (self.pixel_chunk_duration - 1) // self.temporal_compression_factor + 1

    @property
    def temporal_compression_factor(self):
        return self._temporal_compress_factor

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        if num_pixel_frames == 1:
            return 1
        assert num_pixel_frames % self.pixel_chunk_duration == 0, (
            f"Temporal dimension {num_pixel_frames} is not divisible by chunk_length "
            f"{self.pixel_chunk_duration}"
        )
        return (
            num_pixel_frames // self.pixel_chunk_duration * self.latent_chunk_duration
        )

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        if num_latent_frames == 1:
            return 1
        assert num_latent_frames % self.latent_chunk_duration == 0, (
            f"Temporal dimension {num_latent_frames} is not divisible by chunk_length "
            f"{self.latent_chunk_duration}"
        )
        return (
            num_latent_frames // self.latent_chunk_duration * self.pixel_chunk_duration
        )
