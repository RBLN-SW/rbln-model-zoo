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
from typing import Union

from cosmos_transfer1.checkpoints import COSMOS_UPSAMPLER_CHECKPOINT
from cosmos_transfer1.utils.misc import extract_video_frames
from huggingface_hub import snapshot_download
from optimum.rbln import RBLNLlavaForConditionalGeneration
from PIL import Image
from transformers import AutoProcessor, PixtralImageProcessor, PixtralProcessor

from .upsampler_converter import convert_mistral_model, convert_tekken_tokenizer


class RBLNPixtralPromptUpsampler:
    def __init__(self, checkpoint_dir: str, export=False, rbln_config=None):
        """
        Initializes the Upsampler model.
        Args:
            checkpoint_dir (str): The directory where model checkpoints are stored.
            offload_prompt_upsampler (bool, optional):
                If True, the upsampler model will not be loaded during initialization.
                Defaults to False.
        """
        self.checkpoint_dir = checkpoint_dir
        if export:
            self._init_upsampler_model(rbln_config)
        else:
            self._load_upsampler_model(rbln_config)

    def _convert_upsampler_model(self):
        model_path = snapshot_download(
            repo_id=COSMOS_UPSAMPLER_CHECKPOINT, cache_dir=self.checkpoint_dir
        )
        converted_model_path = model_path + "-Converted"
        if not os.path.exists(converted_model_path):
            convert_mistral_model(input_dir=model_path, output_dir=converted_model_path)
        return converted_model_path

    def _init_upsampler_model(self, rbln_config=None):
        """
        Initializes the upsampler model.
        Sets:
            self.upsampler_model:
                An instance of VLM initialized with the specified model configuration.
            self.sampling_params: An instance of SamplingParams with predefined parameters.
        """
        converted_model_path = self._convert_upsampler_model()
        if rbln_config is not None:
            vision_tower_config = rbln_config.get("vision_tower", {})
            language_model_config = rbln_config.get("language_model", {})
        else:
            vision_tower_config = {}
            language_model_config = {}

        rbln_config = {
            "vision_tower": {
                "batch_size": 1,
                "output_hidden_states": True,
                **vision_tower_config,
            },
            "language_model": {
                "num_devices": 4,
                "use_inputs_embeds": True,
                "batch_size": 1,
                "max_seq_len": 32768,
                **language_model_config,
            },
        }

        self.upsampler_model = RBLNLlavaForConditionalGeneration.from_pretrained(
            converted_model_path, export=True, rbln_config=rbln_config
        )
        tokenizer = convert_tekken_tokenizer(
            os.path.join(converted_model_path, "tekken.json")
        )
        image_processor = PixtralImageProcessor()
        self.processor = PixtralProcessor(
            tokenizer=tokenizer, image_processor=image_processor, image_token="[IMG]"
        )

    def _load_upsampler_model(self, rbln_config=None):
        """
        Loads the upsampler model.
        Sets:
            self.upsampler_model:
                An instance of VLM initialized with the specified model configuration.
            self.sampling_params: An instance of SamplingParams with predefined parameters.
        """
        self.upsampler_model = RBLNLlavaForConditionalGeneration.from_pretrained(
            os.path.join(self.checkpoint_dir, "upsampler_model"),
            export=False,
            rbln_config=rbln_config,
        )
        self.processor = AutoProcessor.from_pretrained(
            os.path.join(self.checkpoint_dir, "upsampler_tokenizer")
        )

    def _prompt_upsample(self, prompt: str, video_path: Union[list, str]):
        """
        Generates an upsampled image based on the provided prompt and image paths.
        Args:
            prompt (str): The textual prompt to guide the upsampling process.
            image_paths (list of str): List of file paths to the images to be upsampled.
        Returns:
            str: The text output from the language model after processing the prompt and images.
        """
        prompt = prompt if prompt else "describe the following images"

        image_paths = video_path
        if isinstance(video_path, str):
            image_paths = extract_video_frames(video_path)
        images = [Image.open(frame_path) for frame_path in image_paths]
        text = "<s>[INST]" + prompt + "\n" + "[IMG]" * len(image_paths) + "[/INST]"
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True
        )
        output = self.upsampler_model.generate(
            **inputs,
            max_new_tokens=300,
        )
        input_len = inputs.input_ids.shape[-1]
        output = self.processor.decode(output[0][input_len:], skip_special_tokens=True)

        return str(output).strip()

    def save_pretrained(self, save_directory):
        self.upsampler_model.save_pretrained(
            os.path.join(save_directory, "upsampler_model")
        )
        self.processor.save_pretrained(
            os.path.join(save_directory, "upsampler_tokenizer")
        )
