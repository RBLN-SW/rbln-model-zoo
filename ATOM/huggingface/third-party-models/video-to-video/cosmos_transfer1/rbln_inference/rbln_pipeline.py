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
from typing import Any, Dict, List, Optional, Union

import cv2
import einops
import numpy as np
import rebel
import set_environments  # noqa: F401
import torch
from addict import Dict as Dict_
from cosmos_tokenize1.convert_graph import (
    VAE_DECODER_PATH,
    VAE_ENCODER_PATH,
    optimize_module_by_path,
)
from cosmos_transfer1.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_transfer1.checkpoints import (
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH,
    BASE_7B_CHECKPOINT_PATH,
    COSMOS_TOKENIZER_CHECKPOINT,
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    EDGE2WORLD_CONTROLNET_7B_DISTILLED_CHECKPOINT_PATH,
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    T5_MODEL_CHECKPOINT,
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH,
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH,
    BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH,
    SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH,
    SV2MV_t2w_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SV2MV_v2w_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SV2MV_v2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH,
    SV2MV_v2w_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
)
from cosmos_transfer1.diffusion.config.transfer.config import make_config
from cosmos_transfer1.diffusion.inference.inference_utils import (
    VIDEO_RES_SIZE_INFO,
    detect_aspect_ratio,
    generate_control_input,
    generate_world_from_control,
    get_batched_ctrl_batch,
    get_ctrl_batch_mv,
    get_upscale_size,
    get_video_batch_for_multiview_model,
    load_model_by_config,
    merge_patches_into_video,
    non_strict_load_model,
    read_video_or_image_into_frames_BCTHW,
    resize_control_weight_map,
    resize_video,
    skip_init_linear,
    split_video_into_patches,
    valid_hint_keys,
)
from cosmos_transfer1.diffusion.model.model_ctrl import (
    VideoDiffusionModelWithCtrl,
    VideoDiffusionT2VModelWithCtrl,
    VideoDistillModelWithCtrl,
)
from cosmos_transfer1.diffusion.model.model_multi_camera_ctrl import (
    MultiVideoDiffusionModelWithCtrl,
)
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.config_helper import override
from cosmos_transfer1.utils.lazy_config import instantiate as lazy_instantiate
from cosmos_transfer1.utils.regional_prompting_utils import prepare_regional_prompts
from optimum.rbln.diffusers.pipelines.cosmos.cosmos_guardrail import (
    RBLNCosmosSafetyChecker,
)
from rbln_auxiliary.t5_text_encoder import RBLNCosmosT5TextEncoder
from rbln_auxiliary.upsampler import RBLNPixtralPromptUpsampler
from rbln_network import (
    ControlNetWrapper,
    GeneralDITWrapperWithoutRegion,
    GeneralDITWrapperWithRegion,
    RBLNGeneralDIT,
    RBLNGeneralDITEncoder,
    RBLNGeneralDITMultiviewEncoder,
    RBLNMultiViewVideoExtendGeneralDIT,
    RBLNRuntimeControlNet,
    RBLNRuntimeControlNetMultiview,
    RBLNVideoExtendGeneralDIT,
)
from tqdm import tqdm
from utils.runtime_utils import RBLNRuntimeVAE

MODEL_NAME_DICT = {
    BASE_7B_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3"
    ),
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3"
    ),
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_vis_block3"
    ),
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_depth_block3"
    ),
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_keypoint_block3"
    ),
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_seg_block3"
    ),
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_upscale_block3"
    ),
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH: (
        "CTRL_7Bv1pt3_t2v_121frames_control_input_hdmap_block3"
    ),
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_t2v_121frames_control_input_hdmap_block3"
    ),
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_t2v_121frames_control_input_lidar_block3"
    ),
    BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH: (
        "CTRL_7Bv1pt3_sv2mv_t2w_57frames_control_input_hdmap_block3"
    ),
    BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH: (
        "CTRL_7Bv1pt3_sv2mv_v2w_57frames_control_input_hdmap_block3"
    ),
    SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_sv2mv_t2w_57frames_control_input_hdmap_block3"
    ),
    SV2MV_t2w_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_sv2mv_t2w_57frames_control_input_lidar_block3"
    ),
    SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_sv2mv_t2w_57frames_control_input_hdmap_waymo_block3"
    ),
    SV2MV_v2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_sv2mv_v2w_57frames_control_input_hdmap_waymo_block3"
    ),
    EDGE2WORLD_CONTROLNET_7B_DISTILLED_CHECKPOINT_PATH: (
        "CTRL_7Bv1pt3_lvg_fsdp_distilled_121frames_control_input_edge_block3"
    ),
}
MODEL_CLASS_DICT = {
    BASE_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH: VideoDiffusionT2VModelWithCtrl,
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionT2VModelWithCtrl,
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionT2VModelWithCtrl,
    BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH: MultiVideoDiffusionModelWithCtrl,
    SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: MultiVideoDiffusionModelWithCtrl,
    SV2MV_t2w_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: MultiVideoDiffusionModelWithCtrl,
    BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH: MultiVideoDiffusionModelWithCtrl,
    SV2MV_v2w_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: MultiVideoDiffusionModelWithCtrl,
    SV2MV_v2w_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: MultiVideoDiffusionModelWithCtrl,
    SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH: MultiVideoDiffusionModelWithCtrl,
    SV2MV_v2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH: MultiVideoDiffusionModelWithCtrl,
    EDGE2WORLD_CONTROLNET_7B_DISTILLED_CHECKPOINT_PATH: VideoDistillModelWithCtrl,
}


class RBLNDiffusionControl2WorldGenerationPipeline:
    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_name: str,
        rbln_dir: str = None,
        has_text_input: bool = True,
        guidance: float = 7.0,
        num_steps: int = 35,
        batch_size: int = 1,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: int = 0,
        num_input_frames: int = 1,
        control_inputs: dict = None,
        sigma_max: float = 70.0,
        blur_strength: str = "medium",
        canny_threshold: str = "medium",
        upsample_prompt: bool = False,
        offload_prompt_upsampler: bool = False,
        regional_prompts: List[str] = None,
        region_definitions: Union[List[List[float]], torch.Tensor] = None,
        is_av_sample: bool = False,
        disable_guardrail: bool = False,
        rbln_config: Dict = None,
        export: bool = False,
        create_runtimes: bool = True,
        num_regions: int = None,
    ):
        """Initialize diffusion world generation pipeline.

        Args:
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_guardrail_models: Whether to offload guardrail models
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            seed: Random seed for sampling
            num_input_frames: Number of latent conditions
            control_inputs: Dictionary of control inputs for guided generation
            sigma_max: Sigma max for partial denoising
            blur_strength: Strength of blur applied to input
            canny_threshold: Threshold for edge detection
            upsample_prompt: Whether to upsample prompts using prompt upsampler model
            offload_prompt_upsampler: Whether to offload prompt upsampler after use
            waymo_example: Whether to use the waymo example post-training checkpoint
            disable_guardrail: Whether to disable guardrail checks
            rbln_config:
            export: If True, the model will be compiled. If False, a compiled model will be loaded.
        """
        self.num_input_frames = num_input_frames
        self.control_inputs = control_inputs
        self.sigma_max = sigma_max
        self.blur_strength = blur_strength
        self.canny_threshold = canny_threshold
        self.upsample_prompt = upsample_prompt
        self.offload_prompt_upsampler = offload_prompt_upsampler
        self.prompt_upsampler = None
        self.upsampler_hint_key = None
        self.hint_details = None
        self.model_name = MODEL_NAME_DICT[checkpoint_name]
        self.model_class = MODEL_CLASS_DICT[checkpoint_name]
        self.guidance = guidance
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.fps = fps
        self.num_video_frames = num_video_frames
        self.seed = seed
        self.regional_prompts = regional_prompts
        self.region_definitions = region_definitions
        self.is_av_sample = is_av_sample

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.has_text_input = has_text_input
        self.disable_guardrail = disable_guardrail
        self.rbln_dir = rbln_dir
        self.create_runtimes = create_runtimes
        self.num_regions = num_regions

        config = make_config()
        config = override(config, ["--", f"experiment={self.model_name}"])
        config.validate()
        config.freeze()

        if export:
            if self.regional_prompts:
                assert self.num_regions is not None, (
                    "If you want to compile models that using regional prompts, "
                    "the argument `num_regions` should be provided."
                )

            self._input_shape_checker()
            self._init_text_encoder_model(
                rbln_config=rbln_config.get("text_encoder", None)
            )
            self._init_model(rbln_config=rbln_config.get("transformer", None))
            self._init_network(rbln_config=rbln_config.get("ctrlnet", None))
            self._init_tokenizer(
                config.model.tokenizer, rbln_config=rbln_config.get("vae", {})
            )

            # Initialize prompt upsampler & guardrail models if needed
            if self.upsample_prompt:
                self._init_prompt_upsampler(
                    rbln_config=rbln_config.get("prompt_upsampler", None)
                )
            if not self.disable_guardrail:
                self._init_guardrail_models(
                    rbln_config=rbln_config.get("safety_checker", None)
                )
        else:
            self._load_text_encoder_model(
                rbln_config=rbln_config.get("text_encoder", None)
            )
            self._load_model(rbln_config=rbln_config.get("transformer", None))
            self._load_network(rbln_config=rbln_config.get("ctrlnet", {}))
            self._load_tokenizer(
                config.model.tokenizer, rbln_config=rbln_config.get("vae", {})
            )

            # Load prompt upsampler & guardrail models if needed
            if self.upsample_prompt:
                self._load_prompt_upsampler(
                    rbln_config=rbln_config.get("prompt_upsampler", None)
                )
            if not self.disable_guardrail:
                self._load_guardrail_models(
                    rbln_config=rbln_config.get("safety_checker", None)
                )

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        # Save the text encoder and tokenizer
        try:
            text_encoder_path = os.path.join(save_directory, "text_encoder")
            text_encoder_toknizer_path = os.path.join(save_directory, "tokenizer")
            self.text_encoder.text_encoder.save_pretrained(text_encoder_path)
            self.text_encoder.tokenizer.save_pretrained(text_encoder_toknizer_path)
        except Exception:
            log.warning("Text encoder cannot be save.")

        # Save Cosmos base model (transformer)
        try:
            transformer_path = os.path.join(save_directory, "transformer.rbln")
            self.compiled_model.save(transformer_path)
        except Exception:
            log.warning("Cosmos base model cannot be save.")

        # Save Cosmos ControlNet
        try:
            ctrl_path = os.path.join(save_directory, "ctrlnet")
            os.makedirs(ctrl_path, exist_ok=True)
            for key, ctrlnet in self.compiled_controlnet.items():
                ctrlnet.save(os.path.join(ctrl_path, f"{key}.rbln"))
        except Exception:
            log.warning("ControlNet(s) cannot be save.")

        # Save Cosmos tokenizer (VAE)
        try:
            tokenizer_path = os.path.join(save_directory, "vae")
            os.makedirs(tokenizer_path, exist_ok=True)
            self.vae_decoder_mod.save(os.path.join(tokenizer_path, "decoder.rbln"))
            self.vae_encoder_mod.save(os.path.join(tokenizer_path, "encoder.rbln"))
        except Exception:
            log.warning("Cosmos Tokenizer cannot be save.")

        # Save Prompt upsampler
        try:
            if hasattr(self, "prompt_upsampler"):
                upsampler_path = os.path.join(save_directory, "prompt_upsampler")
                self.prompt_upsampler.save_pretrained(upsampler_path)
        except Exception:
            log.warning("Prompt Upsampler cannot be save.")

        # Save cosmos guardrail models
        try:
            if hasattr(self, "guardrail"):
                safety_checker_path = os.path.join(save_directory, "safety_checker")
                self.guardrail.save_pretrained(safety_checker_path)
        except Exception:
            log.warning("Guardrail models cannot be save.")

        log.info(f"Pipeline submodules are saved to {save_directory}")

    def _input_shape_checker(self):
        # NOTE: This function must be modified if the model input resolution
        # constraints are changed.
        if self.regional_prompts:
            constraint_shapes = [(704, 1280)]
        else:
            constraint_shapes = VIDEO_RES_SIZE_INFO.values()

        assert (self.height, self.width) in constraint_shapes, (
            f"Input shape ({self.height}, {self.width}) is not in supported shapes: "
            f"{list(constraint_shapes)}"
        )

        log.info(
            f"Verified input height and width for compilation: ({self.height}, {self.width})"
        )

    def _init_prompt_upsampler(self, rbln_config=None):
        """
        Initializes the prompt upsampler based on the provided control inputs.

        Returns:
            None: Sets instance variables for prompt upsampler, hint key, and hint details
        """
        log.info("Compile prompt upsampler")
        if rbln_config is None:
            rbln_config = {"create_runtimes": self.create_runtimes}
        else:
            rbln_config["create_runtimes"] = self.create_runtimes
        self.prompt_upsampler = RBLNPixtralPromptUpsampler(
            checkpoint_dir=self.checkpoint_dir,
            export=True,
            rbln_config=rbln_config,
        )

    def _load_prompt_upsampler(self, rbln_config=None):
        vis_hint_keys = ["vis", "edge"]
        other_hint_keys = ["seg", "depth"]
        self.hint_details = None

        log.info("Initializing prompt upsampler...")

        if any(key in vis_hint_keys for key in self.control_inputs):
            self.upsampler_hint_key = "vis"
            self.hint_details = "vis" if "vis" in self.control_inputs else "edge"
        elif any(key in other_hint_keys for key in self.control_inputs):
            selected_hint_keys = [
                key for key in self.control_inputs if key in other_hint_keys
            ]
            self.upsampler_hint_key = selected_hint_keys[0]
        else:
            self.upsampler_hint_key = None

        if self.upsampler_hint_key:
            if rbln_config is None:
                rbln_config = {"create_runtimes": self.create_runtimes}
            else:
                rbln_config["create_runtimes"] = self.create_runtimes
            self.prompt_upsampler = RBLNPixtralPromptUpsampler(
                checkpoint_dir=os.path.join(self.rbln_dir, "prompt_upsampler"),
                export=False,
                rbln_config=rbln_config,
            )

        log.info(
            f"Prompt upsampler initialized with hint key: {self.upsampler_hint_key} "
            f"and hint details: {self.hint_details}"
        )

    def _process_prompt_upsampler(self, prompt, video_path, save_folder):
        """
        Processes and upscales a given prompt using the prompt upsampler.

        Args:
            prompt: The text prompt to upsample
            video_path: Path to the input video
            save_folder: Folder to save intermediate files

        Returns:
            str: The upsampled prompt
        """
        if not self.prompt_upsampler:
            return prompt

        log.info(f"Upsampling prompt with controlnet: {self.upsampler_hint_key}")

        if self.upsampler_hint_key in [
            "vis"
        ]:  # input video or control input, one of them is required
            # prompt upsampler for viscontrol(vis, edge)
            if (
                self.control_inputs[self.hint_details].get("input_control", None)
                is not None
            ):
                input_control_path = self.control_inputs[self.hint_details].get(
                    "input_control", None
                )
            else:
                hint_key = f"control_input_{self.hint_details}"
                input_control_path = generate_control_input(
                    input_file_path=video_path,
                    save_folder=save_folder,
                    hint_key=hint_key,
                    blur_strength=self.blur_strength,
                    canny_threshold=self.canny_threshold,
                )
        else:
            # prompt upsampler for seg, depth
            input_control_path = self.control_inputs[self.upsampler_hint_key].get(
                "input_control", None
            )

        prompt = self.prompt_upsampler._prompt_upsample(
            prompt=prompt, video_path=input_control_path
        )
        return prompt

    def _init_guardrail_models(self, rbln_config=None):
        log.info("Compile guardrail models")
        if rbln_config is None:
            rbln_config = {"create_runtimes": self.create_runtimes}
        else:
            rbln_config["create_runtimes"] = self.create_runtimes

        upscale_factor = 3 if rbln_config.pop("is_upscale", False) else 1
        height = self.height * upscale_factor
        width = self.width * upscale_factor
        self.guardrail = RBLNCosmosSafetyChecker.from_pretrained(
            "nvidia/Cosmos-1.0-Guardrail",
            export=True,
            rbln_config=rbln_config,
            rbln_image_size=(height, width),
        )

    def _load_guardrail_models(self, rbln_config=None):
        if rbln_config is None:
            rbln_config = {"create_runtimes": self.create_runtimes}
        else:
            rbln_config["create_runtimes"] = self.create_runtimes
        self.guardrail = RBLNCosmosSafetyChecker.from_pretrained(
            checkpoint_id=self.rbln_dir,
            subfolder="safety_checker",
            export=False,
            rbln_config=rbln_config,
        )

    def _init_text_encoder_model(self, rbln_config):
        log.info("Compile text encoder model")
        if rbln_config is None:
            rbln_config = {"create_runtimes": self.create_runtimes}
        else:
            rbln_config["create_runtimes"] = self.create_runtimes
        self.text_encoder = RBLNCosmosT5TextEncoder(
            checkpoint_id=T5_MODEL_CHECKPOINT,
            cache_dir=self.checkpoint_dir,
            rbln_config=rbln_config,
            export=True,
        )

    def _load_text_encoder_model(self, rbln_config):
        if rbln_config is None:
            rbln_config = {"create_runtimes": self.create_runtimes}
        else:
            rbln_config["create_runtimes"] = self.create_runtimes
        self.text_encoder = RBLNCosmosT5TextEncoder(
            checkpoint_id=self.rbln_dir,
            rbln_config=rbln_config,
            export=False,
        )

    def _init_model(self, rbln_config=None, is_multiview=False):
        log.info("Compile base model")
        model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos-transfer1/cosmos_transfer1/diffusion/config/transfer/config.py",
            model_class=self.model_class,
            base_checkpoint_dir=self.checkpoint_dir,
        )
        model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
        ckpt_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)

        with skip_init_linear():
            model.set_up_model()
        net_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        non_strict_load_model(model.model, net_state_dict)

        hidden_size = model.model.base_model.net.model_channels
        attention_head_dim = hidden_size // model.model.base_model.net.num_heads
        spatial_compression_factor = (
            model.config.tokenizer.video_vae.spatial_compression_factor
        )
        temporal_compression_factor = (
            model.config.tokenizer.video_vae.temporal_compression_factor
        )
        patch_spatial = model.model.base_model.net.patch_spatial
        patch_temporal = model.model.base_model.net.patch_temporal

        latent_height = self.height // spatial_compression_factor
        latent_width = self.width // spatial_compression_factor
        num_latent_frames = (
            self.num_video_frames - 1
        ) // temporal_compression_factor + 1

        n_views = (
            1 if not hasattr(model.model.net, "n_views") else model.model.net.n_views
        )
        hidden_dim = (
            (latent_height // patch_spatial)
            * (latent_width // patch_spatial)
            * num_latent_frames
            * n_views
            // patch_temporal
        )

        # TODO: set input info
        input_info = [
            (
                "x",
                [
                    num_latent_frames * n_views,
                    latent_height // patch_spatial,
                    latent_width // patch_spatial,
                    self.batch_size,
                    hidden_size,
                ],
                torch.float32,
            ),
            ("affline_emb_B_D", [self.batch_size, hidden_size], torch.float32),
            ("crossattn_emb", [512 * n_views, self.batch_size, 1024], torch.float32),
            ("cos_", [hidden_dim, 1, 1, attention_head_dim], torch.float32),
            ("sin_", [hidden_dim, 1, 1, attention_head_dim], torch.float32),
            ("adaln_lora_B_3D", [self.batch_size, hidden_size * 3], torch.float32),
            (
                "extra_per_block_pos_emb",
                [
                    num_latent_frames * n_views,
                    latent_height // patch_spatial,
                    latent_width // patch_spatial,
                    self.batch_size,
                    hidden_size,
                ],
                torch.float32,
            ),
        ]

        original_shape = torch.Size(
            [
                self.batch_size,
                17,
                num_latent_frames * n_views,
                latent_height,
                latent_width,
            ]
        )
        if self.regional_prompts:
            wrapped_model = GeneralDITWrapperWithRegion(
                model.model.base_model.net, original_shape
            )
            input_info.extend(
                [
                    (
                        "regional_contexts",
                        [self.num_regions, 512, self.batch_size, 1024],
                        torch.float32,
                    ),
                    (
                        "region_masks",
                        [
                            self.num_regions,
                            self.num_video_frames,
                            latent_height,
                            latent_width,
                            self.batch_size,
                        ],
                        torch.float32,
                    ),
                    ("base_ratio", [1], torch.float32),
                ]
            )
        else:
            wrapped_model = GeneralDITWrapperWithoutRegion(
                model.model.base_model.net, original_shape
            )

        num_ctrl_blocks = len(model.config.net_ctrl.layer_mask) - sum(
            model.config.net_ctrl.layer_mask
        )
        for i in range(num_ctrl_blocks):
            input_info.append(
                (
                    f"x_ctrl_{i}",
                    [
                        num_latent_frames * n_views,
                        latent_height // patch_spatial,
                        latent_width // patch_spatial,
                        self.batch_size,
                        hidden_size,
                    ],
                    torch.float32,
                )
            )

        self.compiled_model = rebel.compile_from_torch(
            wrapped_model,
            input_info,
            num_devices=rbln_config.get("num_devices", None),
        )
        if is_multiview:
            runtime = RBLNMultiViewVideoExtendGeneralDIT(
                self.compiled_model,
                model.model.base_model.net,
                rbln_config,
                self.num_regions,
            )
        else:
            if self.is_av_sample:
                runtime = RBLNGeneralDIT(
                    self.compiled_model,
                    model.model.base_model.net,
                    rbln_config,
                    self.num_regions,
                )
            else:
                runtime = RBLNVideoExtendGeneralDIT(
                    self.compiled_model,
                    model.model.base_model.net,
                    rbln_config,
                    self.num_regions,
                )
        if self.create_runtimes:
            runtime.create_runtime()

        model_dict = Dict_()
        model_dict.net = model.model.net
        model_dict.conditioner = model.model.conditioner
        model_dict.base_model = Dict_()
        model_dict.base_model.net = runtime
        model_dict.base_model.conditioner = model.model.base_model.conditioner
        model_dict.base_model.logvar = model.model.base_model.logvar
        model_dict.logvar = model.logvar
        delattr(model, "model")
        setattr(model, "model", model_dict)
        self.model = model

    def _load_model(self, rbln_config=None, is_multiview=False):
        model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos-transfer1/cosmos_transfer1/diffusion/config/transfer/config.py",
            model_class=self.model_class,
            base_checkpoint_dir=self.checkpoint_dir,
        )
        model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
        ckpt_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)

        with skip_init_linear():
            model.set_up_model()
        net_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        non_strict_load_model(model.model, net_state_dict)

        self.compiled_model = rebel.RBLNCompiledModel(
            os.path.join(self.rbln_dir, "transformer.rbln")
        )
        num_regions = (
            len(self.regional_prompts)
            if isinstance(self.regional_prompts, (tuple, list))
            else None
        )
        if is_multiview:
            runtime = RBLNMultiViewVideoExtendGeneralDIT(
                self.compiled_model,
                model.model.base_model.net,
                rbln_config,
                num_regions,
            )
        else:
            if self.is_av_sample:
                runtime = RBLNGeneralDIT(
                    self.compiled_model,
                    model.model.base_model.net,
                    rbln_config,
                    num_regions,
                )
            else:
                runtime = RBLNVideoExtendGeneralDIT(
                    self.compiled_model,
                    model.model.base_model.net,
                    rbln_config,
                    num_regions,
                )
        if self.create_runtimes:
            runtime.create_runtime()
        model_dict = Dict_()
        model_dict.net = model.model.net
        model_dict.conditioner = model.model.conditioner
        model_dict.base_model = Dict_()
        model_dict.base_model.net = runtime
        model_dict.base_model.conditioner = model.model.base_model.conditioner
        model_dict.base_model.logvar = model.model.base_model.logvar
        model_dict.logvar = model.logvar
        delattr(model, "model")
        setattr(model, "model", model_dict)
        self.model = model

    def _init_network(self, rbln_config=None, is_multiview=False):
        log.info("Compile ctrl models")
        hint_encoders = {}
        compiled_controlnet = {}
        for key, spec in self.control_inputs.items():
            if key in valid_hint_keys:
                model = load_model_by_config(
                    config_job_name=self.model_name,
                    config_file="cosmos-transfer1/cosmos_transfer1/diffusion/config/transfer/config.py",
                    model_class=self.model_class,
                    base_checkpoint_dir=self.checkpoint_dir,
                )
                n_views = 1 if not hasattr(model, "n_views") else model.n_views

                model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
                ckpt_path = spec["ckpt_path"]

                log.info(f"Loading ctrl model from ckpt_path: {spec['ckpt_path']}")
                with skip_init_linear():
                    model.set_up_model()
                net_state_dict = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                non_strict_load_model(model.model, net_state_dict)

                hidden_size = model.model.base_model.net.model_channels
                attention_head_dim = hidden_size // model.model.base_model.net.num_heads
                spatial_compression_factor = (
                    model.config.tokenizer.video_vae.spatial_compression_factor
                )
                temporal_compression_factor = (
                    model.config.tokenizer.video_vae.temporal_compression_factor
                )
                patch_spatial = model.model.base_model.net.patch_spatial
                patch_temporal = model.model.base_model.net.patch_temporal

                latent_height = self.height // spatial_compression_factor
                latent_width = self.width // spatial_compression_factor
                num_latent_frames = (
                    self.num_video_frames - 1
                ) // temporal_compression_factor + 1

                hidden_dim = (
                    (latent_height // patch_spatial)
                    * (latent_width // patch_spatial)
                    * num_latent_frames
                    * n_views
                    // patch_temporal
                )
                input_info = [
                    (
                        "guided_hints",
                        [
                            num_latent_frames * n_views,
                            latent_height // patch_spatial,
                            latent_width // patch_spatial,
                            self.batch_size,
                            hidden_size,
                        ],
                        torch.float32,
                    ),
                    (
                        "x",
                        [
                            num_latent_frames * n_views,
                            latent_height // patch_spatial,
                            latent_width // patch_spatial,
                            self.batch_size,
                            hidden_size,
                        ],
                        torch.float32,
                    ),
                    ("cos_", [hidden_dim, 1, 1, attention_head_dim], torch.float32),
                    ("sin_", [hidden_dim, 1, 1, attention_head_dim], torch.float32),
                    (
                        "extra_pos_emb",
                        [
                            num_latent_frames * n_views,
                            latent_height // patch_spatial,
                            latent_width // patch_spatial,
                            self.batch_size,
                            hidden_size,
                        ],
                        torch.float32,
                    ),
                    (
                        "crossattn_emb",
                        [512 * n_views, self.batch_size, 1024],
                        torch.float32,
                    ),
                    (
                        "adaln_lora_B_3D",
                        [self.batch_size, hidden_size * 3],
                        torch.float32,
                    ),
                    ("affline_emb_B_D", [self.batch_size, hidden_size], torch.float32),
                    ("control_weight", [1], torch.float32),
                ]
                if self.regional_prompts:
                    input_info.extend(
                        [
                            (
                                "regional_contexts",
                                [self.num_regions, 512, self.batch_size, 1024],
                                torch.float32,
                            ),
                            (
                                "region_masks",
                                [
                                    self.num_regions,
                                    self.num_video_frames,
                                    latent_height,
                                    latent_width,
                                    self.batch_size,
                                ],
                                torch.float32,
                            ),
                            ("base_ratio", [1], torch.float32),
                        ]
                    )
                wrapped_controlnet = ControlNetWrapper(model.model.net).eval()
                compiled_model = rebel.compile_from_torch(
                    wrapped_controlnet,
                    input_info,
                    num_devices=rbln_config.get(key, {}).get("num_devices", None),
                )
                compiled_controlnet[key] = compiled_model
                if is_multiview:
                    controlnet_runtime = RBLNRuntimeControlNetMultiview(
                        compiled_model,
                        model.model.net,
                        rbln_config.get(key, {}),
                        self.num_regions,
                    )
                else:
                    controlnet_runtime = RBLNRuntimeControlNet(
                        compiled_model,
                        model.model.net,
                        rbln_config.get(key, {}),
                        self.num_regions,
                    )
                if self.create_runtimes:
                    controlnet_runtime.create_runtime()

                hint_encoders[key] = controlnet_runtime
        in_channels = model.model.net.in_channels
        use_cross_attn_mask = model.model.net.use_cross_attn_mask
        del model
        self.compiled_controlnet = compiled_controlnet
        if is_multiview:
            self.model.model.net = RBLNGeneralDITMultiviewEncoder(
                hint_encoders,
                self.control_inputs.keys(),
                in_channels,
                use_cross_attn_mask,
            )
        else:
            self.model.model.net = RBLNGeneralDITEncoder(
                hint_encoders,
                self.control_inputs.keys(),
                in_channels,
                use_cross_attn_mask,
            )

    def _load_network(self, rbln_config=None, is_multiview=False):
        """
        load the hint encoders. these encoders are run along with the main model
        to provide additional context
        """
        hint_encoders = {}
        compiled_controlnet = {}
        num_regions = (
            len(self.regional_prompts)
            if isinstance(self.regional_prompts, (tuple, list))
            else None
        )
        for key, spec in self.control_inputs.items():
            if key in valid_hint_keys:
                model = load_model_by_config(
                    config_job_name=self.model_name,
                    config_file="cosmos-transfer1/cosmos_transfer1/diffusion/config/transfer/config.py",
                    model_class=self.model_class,
                    base_checkpoint_dir=self.checkpoint_dir,
                )

                model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
                ckpt_path = spec["ckpt_path"]

                log.info(f"Loading ctrl model from ckpt_path: {spec['ckpt_path']}")
                with skip_init_linear():
                    model.set_up_model()
                net_state_dict = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                non_strict_load_model(model.model, net_state_dict)

                compiled_controlnet[key] = rebel.RBLNCompiledModel(
                    os.path.join(self.rbln_dir, "ctrlnet", f"{key}.rbln")
                )
                if is_multiview:
                    controlnet_runtime = RBLNRuntimeControlNetMultiview(
                        compiled_controlnet[key],
                        model.model.net,
                        rbln_config.get(key, {}),
                        num_regions,
                    )
                else:
                    controlnet_runtime = RBLNRuntimeControlNet(
                        compiled_controlnet[key],
                        model.model.net,
                        rbln_config.get(key, {}),
                        num_regions,
                    )

                if self.create_runtimes:
                    controlnet_runtime.create_runtime()

                hint_encoders[key] = controlnet_runtime
        in_channels = model.model.net.in_channels
        use_cross_attn_mask = model.model.net.use_cross_attn_mask
        del model
        self.compiled_controlnet = compiled_controlnet
        if is_multiview:
            self.model.model.net = RBLNGeneralDITMultiviewEncoder(
                hint_encoders,
                self.control_inputs.keys(),
                in_channels,
                use_cross_attn_mask,
            )
        else:
            self.model.model.net = RBLNGeneralDITEncoder(
                hint_encoders,
                self.control_inputs.keys(),
                in_channels,
                use_cross_attn_mask,
            )

    def _init_tokenizer(self, config, rbln_config):
        log.info("Compile tokenizer model")

        tokenizer_path = os.path.join(self.checkpoint_dir, COSMOS_TOKENIZER_CHECKPOINT)
        tokenizer = lazy_instantiate(config)
        tokenizer.load_weights(tokenizer_path)
        if hasattr(tokenizer, "reset_dtype"):
            tokenizer.reset_dtype()

        vae_encoder = tokenizer.video_vae.encoder.to(dtype=torch.float32)
        for module in VAE_ENCODER_PATH:
            optimize_module_by_path(vae_encoder, module)

        vae_decoder = tokenizer.video_vae.decoder.to(dtype=torch.float32)
        for module in VAE_DECODER_PATH:
            optimize_module_by_path(vae_decoder, module)

        vae_encoder.encoder.eval()
        vae_decoder.decoder.eval()

        spatial_compression_factor = config.video_vae.spatial_compression_factor
        temporal_compression_factor = config.video_vae.temporal_compression_factor
        num_latent_frames = (
            self.num_video_frames - 1
        ) // temporal_compression_factor + 1

        latent_height = self.height // spatial_compression_factor
        latent_width = self.width // spatial_compression_factor

        # compile Cosmos transfer 1 Autoencoder (vae decoder)
        vae_encoder_mod = rebel.compile_from_torchscript(
            vae_encoder,
            input_info=[
                (
                    "x",
                    [1, 3, self.num_video_frames, self.height, self.width],
                    torch.float32,
                )
            ],
        )
        vae_decoder_mod = rebel.compile_from_torchscript(
            vae_decoder,
            input_info=[
                (
                    "x",
                    [
                        1,
                        config.latent_ch,
                        num_latent_frames,
                        latent_height,
                        latent_width,
                    ],
                    torch.float32,
                )
            ],
        )
        tokenizer.video_vae.max_enc_batch_size = 1
        tokenizer.video_vae.max_dec_batch_size = 1

        encoder_rbln_config = rbln_config.get("encoder", {})
        decoder_rbln_config = rbln_config.get("decoder", {})

        self.vae_encoder_mod = vae_encoder_mod
        self.vae_decoder_mod = vae_decoder_mod
        vae_encoder_runtime = RBLNRuntimeVAE(vae_encoder_mod, encoder_rbln_config)
        vae_decoder_runtime = RBLNRuntimeVAE(vae_decoder_mod, decoder_rbln_config)

        if self.create_runtimes:
            vae_encoder_runtime.create_runtime()
            vae_decoder_runtime.create_runtime()

        delattr(tokenizer.video_vae, "encoder")
        delattr(tokenizer.video_vae, "decoder")
        setattr(tokenizer.video_vae, "encoder", vae_encoder_runtime)
        setattr(tokenizer.video_vae, "decoder", vae_decoder_runtime)
        self.model.tokenizer = tokenizer

    def _load_tokenizer(self, config, rbln_config):
        tokenizer_path = os.path.join(self.checkpoint_dir, COSMOS_TOKENIZER_CHECKPOINT)
        tokenizer = lazy_instantiate(config)
        tokenizer.load_weights(tokenizer_path)
        if hasattr(tokenizer, "reset_dtype"):
            tokenizer.reset_dtype()

        tokenizer.video_vae.max_enc_batch_size = 1
        tokenizer.video_vae.max_dec_batch_size = 1

        rbln_tokenizer_path = os.path.join(self.rbln_dir, "vae")
        encoder_rbln_config = rbln_config.get("encoder", {})
        decoder_rbln_config = rbln_config.get("decoder", {})

        # get height and width from rbln vae encoder metadata
        _metadata = rebel.RBLNCompiledModel.inspect(
            os.path.join(rbln_tokenizer_path, "encoder.rbln")
        )
        self.height = _metadata["inputs"][0]["shape"][3]
        self.width = _metadata["inputs"][0]["shape"][4]

        vae_encoder_mod = rebel.RBLNCompiledModel(
            os.path.join(rbln_tokenizer_path, "encoder.rbln")
        )
        vae_decoder_mod = rebel.RBLNCompiledModel(
            os.path.join(rbln_tokenizer_path, "decoder.rbln")
        )

        vae_encoder_runtime = RBLNRuntimeVAE(vae_encoder_mod, encoder_rbln_config)
        vae_decoder_runtime = RBLNRuntimeVAE(vae_decoder_mod, decoder_rbln_config)

        if self.create_runtimes:
            vae_encoder_runtime.create_runtime()
            vae_decoder_runtime.create_runtime()

        delattr(tokenizer.video_vae, "encoder")
        delattr(tokenizer.video_vae, "decoder")
        setattr(tokenizer.video_vae, "encoder", vae_encoder_runtime)
        setattr(tokenizer.video_vae, "decoder", vae_decoder_runtime)
        self.model.tokenizer = tokenizer

    def _run_guardrail_on_prompt(self, prompt: str) -> bool:
        """Check if prompt meets safety requirements.

        Validates the input prompt against safety policies using loaded guardrail models.

        Args:
            prompt: Raw text prompt to validate

        Returns:
            bool: True if prompt passes all safety checks, False otherwise
        """
        if self.disable_guardrail:
            return True

        return guardrail_presets.run_text_guardrail(
            prompt, self.guardrail.text_guardrail
        )

    def _run_guardrail_on_video(self, video: np.ndarray) -> np.ndarray | None:
        """Check if video meets safety requirements.

        Validates generated video content against safety policies using guardrail models.

        Args:
            video: Video frames to validate

        Returns:
            np.ndarray: Processed video if safe, None if unsafe
        """
        if self.disable_guardrail:
            return video

        return guardrail_presets.run_video_guardrail(
            video, self.guardrail.video_guardrail
        )

    def _run_text_embedding_on_prompt(
        self, prompts: list[str], **kwargs: Any
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Convert text prompts to embeddings.

        Processes text prompts into embedding tensors that condition the generation model.

        Args:
            prompts: List of text prompts to encode
            **kwargs: Additional arguments for text encoding

        Returns:
            tuple containing:
                - List of text embedding tensors for each prompt
                - List of attention masks for each embedding
        """

        embeddings = []
        masks = []
        for prompt in prompts:
            embedding, mask = self.text_encoder.encode_prompts(
                [prompt],
                **kwargs,
            )
            embeddings.append(embedding)
            masks.append(mask)

        return embeddings, masks

    def _run_tokenizer_decoding(
        self, sample: torch.Tensor, use_batch: bool = True
    ) -> np.ndarray:
        """Decode latent samples to video frames using the tokenizer decoder.

        Args:
            sample: Latent tensor from diffusion model [B, C, T, H, W]

        Returns:
            np.ndarray: Decoded video frames as uint8 numpy array [T, H, W, C]
                        with values in range [0, 255]
        """
        # Decode video
        if sample.shape[0] == 1 or use_batch:
            video = (1.0 + self.model.decode(sample)).clamp(0, 2) / 2  # [B, 3, T, H, W]
        else:
            # Do decoding for each batch sequentially to prevent OOM.
            samples = []
            for sample_i in sample:
                samples += [self.model.decode(sample_i.unsqueeze(0)).cpu()]
            samples = (torch.cat(samples) + 1).clamp(0, 2) / 2
            # samples = (torch.stack(samples) + 1).clamp(0, 2) / 2

            # Stitch the patches together to form the final video.
            patch_h, patch_w = samples.shape[-2:]
            orig_size = (patch_w, patch_h)
            aspect_ratio = detect_aspect_ratio(orig_size)
            stitch_w, stitch_h = get_upscale_size(
                orig_size, aspect_ratio, upscale_factor=3
            )
            n_img_w = (stitch_w - 1) // patch_w + 1
            n_img_h = (stitch_h - 1) // patch_h + 1
            overlap_size_w = overlap_size_h = 0
            if n_img_w > 1:
                overlap_size_w = (n_img_w * patch_w - stitch_w) // (n_img_w - 1)
            if n_img_h > 1:
                overlap_size_h = (n_img_h * patch_h - stitch_h) // (n_img_h - 1)
            video = merge_patches_into_video(
                samples, overlap_size_h, overlap_size_w, n_img_h, n_img_w
            )
            video = torch.nn.functional.interpolate(
                video[0], size=(patch_h * 3, patch_w * 3), mode="bicubic"
            )[None]
            video = video.clamp(0, 1)
        video = (video * 255).to(torch.uint8).cpu()
        return video

    def _run_model(
        self,
        prompt_embeddings: torch.Tensor,  # [B, ...]
        video_paths: list[str],  # [B]
        negative_prompt_embeddings: Optional[torch.Tensor] = None,  # [B, ...] or None
        control_inputs_list: list[dict] = None,  # [B] list of dicts
    ) -> np.ndarray:
        """
        Batched world generation with model offloading.
        Each batch element corresponds to a (prompt, video, control_inputs) triple.
        """

        prompt_embeddings = torch.cat(prompt_embeddings)
        if negative_prompt_embeddings is not None:
            negative_prompt_embeddings = torch.cat(negative_prompt_embeddings)

        B = len(video_paths)
        assert prompt_embeddings.shape[0] == B, (
            "Batch size mismatch for prompt embeddings"
        )
        if negative_prompt_embeddings is not None:
            assert negative_prompt_embeddings.shape[0] == B, (
                "Batch size mismatch for negative prompt embeddings"
            )
        assert len(control_inputs_list) == B, (
            "Batch size mismatch for control_inputs_list"
        )

        log.info("Starting data augmentation")

        # Process regional prompts if provided
        log.info(f"regional_prompts passed to _run_model: {self.regional_prompts}")
        log.info(f"region_definitions passed to _run_model: {self.region_definitions}")
        regional_embeddings, _ = self._run_text_embedding_on_prompt(
            self.regional_prompts
        )
        regional_contexts = None
        region_masks = None
        if self.regional_prompts and self.region_definitions:
            # Prepare regional prompts using the existing text embedding function
            _, regional_contexts, region_masks = prepare_regional_prompts(
                model=self.model,
                global_prompt=prompt_embeddings,  # Pass the already computed global embedding
                regional_prompts=regional_embeddings,
                region_definitions=self.region_definitions,
                batch_size=1,  # Adjust based on your batch size
                time_dim=self.num_video_frames,
                height=self.height // self.model.tokenizer.spatial_compression_factor,
                width=self.width // self.model.tokenizer.spatial_compression_factor,
                device=torch.device("cpu"),
                compression_factor=self.model.tokenizer.spatial_compression_factor,
            )

        is_upscale_case = any(
            "upscale" in control_inputs for control_inputs in control_inputs_list
        )
        # Get video batch and state shape
        data_batch, state_shape = get_batched_ctrl_batch(
            model=self.model,
            prompt_embeddings=prompt_embeddings,  # [B, ...]
            negative_prompt_embeddings=negative_prompt_embeddings,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames,
            input_video_paths=video_paths,  # [B]
            control_inputs_list=control_inputs_list,  # [B]
            blur_strength=self.blur_strength,
            canny_threshold=self.canny_threshold,
        )

        if data_batch["input_video"] is None:
            ctrl_video = next(
                (v for k, v in data_batch.items() if k.startswith("control_input")),
                None,
            )
            height, width = ctrl_video.shape[3], ctrl_video.shape[4]
        else:
            input_video = data_batch["input_video"]
            height, width = input_video.shape[3], input_video.shape[4]

        assert height == self.height, (
            f"Resized input video height {height} does "
            f"not match compiled model input height {self.height}, You must "
            f"re-compile the model with the new height that is the same as the "
            f"resized input video height."
        )
        assert width == self.width, (
            f"Resized input video width {width} does "
            f"not match compiled model input width {self.width}, You must "
            f"re-compile the model with the new width that is the same as the "
            f"resized input video width."
        )

        if regional_contexts is not None:
            data_batch["regional_contexts"] = regional_contexts.to(dtype=torch.float32)
            data_batch["region_masks"] = region_masks.to(dtype=torch.float32)

        log.info("Completed data augmentation")

        hint_key = data_batch["hint_key"]
        control_input = data_batch[hint_key]  # [B, C, T, H, W]
        input_video = data_batch.get("input_video", None)
        control_weight = data_batch.get("control_weight", None)
        num_new_generated_frames = self.num_video_frames - self.num_input_frames
        B, C, T, H, W = control_input.shape
        if (
            T - self.num_input_frames
        ) % num_new_generated_frames != 0:  # pad duplicate frames at the end
            pad_t = num_new_generated_frames - (
                (T - self.num_input_frames) % num_new_generated_frames
            )
            pad_frames = control_input[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
            control_input = torch.cat([control_input, pad_frames], dim=2)
            if input_video is not None:
                pad_video = input_video[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
                input_video = torch.cat([input_video, pad_video], dim=2)
            num_total_frames_with_padding = control_input.shape[2]
            if (
                isinstance(control_weight, torch.Tensor)
                and control_weight.ndim > 5
                and num_total_frames_with_padding > control_weight.shape[3]
            ):
                pad_t = num_total_frames_with_padding - control_weight.shape[3]
                pad_weight = control_weight[:, :, :, -1:].repeat(1, 1, 1, pad_t, 1, 1)
                control_weight = torch.cat([control_weight, pad_weight], dim=3)
        else:
            num_total_frames_with_padding = T
        N_clip = (
            num_total_frames_with_padding - self.num_input_frames
        ) // num_new_generated_frames

        video = []
        prev_frames = None
        for i_clip in tqdm(range(N_clip)):
            data_batch_i = {k: v for k, v in data_batch.items()}
            start_frame = num_new_generated_frames * i_clip
            end_frame = num_new_generated_frames * (i_clip + 1) + self.num_input_frames

            # Prepare x_sigma_max
            if input_video is not None:
                if is_upscale_case:
                    x_sigma_max = []
                    for b in range(B):
                        input_frames = input_video[b : b + 1, :, start_frame:end_frame]
                        x0 = self.model.encode(input_frames).contiguous()
                        x_sigma_max.append(
                            self.model.get_x_from_clean(
                                x0, self.sigma_max, seed=(self.seed + i_clip)
                            )
                        )
                    x_sigma_max = torch.cat(x_sigma_max)
                else:
                    input_frames = input_video[:, :, start_frame:end_frame]
                    x0 = self.model.encode(input_frames).contiguous()
                    x_sigma_max = self.model.get_x_from_clean(
                        x0, self.sigma_max, seed=(self.seed + i_clip)
                    )

            else:
                x_sigma_max = None

            data_batch_i[hint_key] = control_input[:, :, start_frame:end_frame]
            latent_hint = []
            log.info("Starting latent encoding")
            for b in range(B):
                data_batch_p = {k: v for k, v in data_batch_i.items()}
                data_batch_p[hint_key] = data_batch_i[hint_key][b : b + 1]
                if len(control_inputs_list) >= 1 and len(control_inputs_list[0]) > 1:
                    latent_hint_i = []
                    for idx in range(0, data_batch_p[hint_key].size(1), 3):
                        x_rgb = data_batch_p[hint_key][:, idx : idx + 3]
                        latent_hint_i.append(self.model.encode(x_rgb))
                    latent_hint.append(torch.cat(latent_hint_i).unsqueeze(0))
                else:
                    latent_hint.append(self.model.encode_latent(data_batch_p))
            data_batch_i["latent_hint"] = latent_hint = torch.cat(latent_hint)
            log.info("Completed latent encoding")
            # Resize control_weight if needed
            if isinstance(control_weight, torch.Tensor) and control_weight.ndim > 4:
                control_weight_t = control_weight[..., start_frame:end_frame, :, :]
                t, h, w = latent_hint.shape[-3:]
                data_batch_i["control_weight"] = resize_control_weight_map(
                    control_weight_t, (t, h // 2, w // 2)
                )

            # Prepare condition_latent for long video generation
            if i_clip == 0:
                num_input_frames = 0
                latent_tmp = latent_hint if latent_hint.ndim == 5 else latent_hint[:, 0]
                condition_latent = torch.zeros_like(latent_tmp)
            else:
                num_input_frames = self.num_input_frames
                prev_frames = split_video_into_patches(
                    prev_frames, control_input.shape[-2], control_input.shape[-1]
                )
                input_frames = prev_frames.float() / 255.0 * 2 - 1
                condition_latent = self.model.encode(input_frames).contiguous()

            # Generate video frames for this clip (batched)
            log.info("Starting diffusion sampling")
            latents = generate_world_from_control(
                model=self.model,
                state_shape=state_shape,
                is_negative_prompt=True,
                data_batch=data_batch_i,
                guidance=self.guidance,
                num_steps=self.num_steps,
                seed=(self.seed + i_clip),
                condition_latent=condition_latent,
                num_input_frames=num_input_frames,
                sigma_max=self.sigma_max if x_sigma_max is not None else None,
                x_sigma_max=x_sigma_max,
                use_batch_processing=False if is_upscale_case else True,
            )
            log.info("Completed diffusion sampling")
            log.info("Starting VAE decode")
            frames = self._run_tokenizer_decoding(
                latents, use_batch=False if is_upscale_case else True
            )  # [B, T, H, W, C] or similar
            log.info("Completed VAE decode")

            if i_clip == 0:
                video.append(frames)
            else:
                video.append(frames[:, :, self.num_input_frames :])

            prev_frames = torch.zeros_like(frames)
            prev_frames[:, :, : self.num_input_frames] = frames[
                :, :, -self.num_input_frames :
            ]

        video = torch.cat(video, dim=2)[:, :, :T]
        video = video.permute(0, 2, 3, 4, 1).numpy()
        return video

    def generate(
        self,
        prompt: str | list[str],
        video_path: str | list[str],
        negative_prompt: Optional[str | list[str]] = None,
        control_inputs: dict | list[dict] = None,
        save_folder: str = "outputs/",
        batch_size: int = 1,
    ) -> tuple[np.ndarray, str | list[str]] | None:
        """Generate video from text prompt and control video.

        Pipeline steps:
        1. Run safety checks on input prompt
        2. Convert prompt to embeddings
        3. Generate video frames using diffusion
        4. Run safety checks and apply face blur on generated video frames

        Args:
            prompt: Text description of desired video
            video_path: Path to input video
            negative_prompt: Optional text to guide what not to generate
            control_inputs: Control inputs for guided generation
            save_folder: Folder to save intermediate files
            batch_size: Number of videos to process simultaneously

        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """

        prompts = [prompt] if isinstance(prompt, str) else prompt
        video_paths = [video_path] if isinstance(video_path, str) else video_path
        control_inputs_list = (
            [control_inputs] if not isinstance(control_inputs, list) else control_inputs
        )

        assert len(video_paths) == batch_size, "Number of prompts and videos must match"
        assert len(control_inputs_list) == batch_size, (
            "Number of control inputs must match batch size"
        )
        log.info(f"Running batch generation with batch_size={batch_size}")

        # Process prompts in batch
        all_videos = []
        all_final_prompts = []

        # Upsample prompts if enabled
        if self.prompt_upsampler:
            upsampled_prompts = []
            for i, (single_prompt, single_video_path) in enumerate(
                zip(prompts, video_paths)
            ):
                log.info(
                    f"Upsampling prompt {i + 1}/{batch_size}: {single_prompt[:50]}..."
                )
                video_save_subfolder = os.path.join(save_folder, f"video_{i}")
                os.makedirs(video_save_subfolder, exist_ok=True)
                upsampled_prompt = self._process_prompt_upsampler(
                    single_prompt, single_video_path, video_save_subfolder
                )
                upsampled_prompts.append(upsampled_prompt)
                log.info(f"Upsampled prompt {i + 1}: {upsampled_prompt[:50]}...")
            prompts = upsampled_prompts

        log.info("Running guardrail checks on all prompts")
        safe_indices = []
        for i, single_prompt in enumerate(prompts):
            is_safe = self._run_guardrail_on_prompt(single_prompt)
            if is_safe:
                safe_indices.append(i)
            else:
                log.critical(f"Input text prompt {i + 1} is not safe")

        if not safe_indices:
            log.critical("All prompts failed safety checks")
            return None

        safe_prompts = [prompts[i] for i in safe_indices]
        safe_video_paths = [video_paths[i] for i in safe_indices]
        safe_control_inputs = [control_inputs_list[i] for i in safe_indices]

        log.info("Running text embedding on all prompts")
        all_prompt_embeddings = []

        # Process in smaller batches if needed to avoid OOM
        embedding_batch_size = min(
            batch_size, 8
        )  # Process embeddings in smaller batches
        for i in range(0, len(safe_prompts), embedding_batch_size):
            batch_prompts = safe_prompts[i : i + embedding_batch_size]
            if negative_prompt:
                batch_prompts_with_neg = []
                for p in batch_prompts:
                    batch_prompts_with_neg.extend([p, negative_prompt])
            else:
                batch_prompts_with_neg = batch_prompts
            log.info("Starting T5 compute")
            prompt_embeddings, _ = self._run_text_embedding_on_prompt(
                batch_prompts_with_neg
            )
            log.info("Completed T5 compute")
            # Separate positive and negative embeddings
            if negative_prompt:
                for j in range(0, len(prompt_embeddings), 2):
                    all_prompt_embeddings.append(
                        (prompt_embeddings[j], prompt_embeddings[j + 1])
                    )
            else:
                for emb in prompt_embeddings:
                    all_prompt_embeddings.append((emb, None))
        log.info("Finish text embedding on prompt")

        # Generate videos in batches
        log.info("Run generation")

        all_neg_embeddings = [emb[1] for emb in all_prompt_embeddings]
        all_prompt_embeddings = [emb[0] for emb in all_prompt_embeddings]
        videos = self._run_model(
            prompt_embeddings=all_prompt_embeddings,
            negative_prompt_embeddings=all_neg_embeddings,
            video_paths=safe_video_paths,
            control_inputs_list=safe_control_inputs,
        )
        log.info("Finish generation")

        log.info("Run guardrail on generated videos")
        for i, video in enumerate(videos):
            safe_video = self._run_guardrail_on_video(video)
            if safe_video is not None:
                all_videos.append(safe_video)
                all_final_prompts.append(safe_prompts[i])
            else:
                log.critical(f"Generated video {i + 1} is not safe")
        if not all_videos:
            log.critical("All generated videos failed safety checks")
            return None
        return all_videos, all_final_prompts


class RBLNDiffusionControl2WorldMultiviewGenerationPipeline(
    RBLNDiffusionControl2WorldGenerationPipeline
):
    def __init__(self, *args, is_lvg_model=False, n_clip_max=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_lvg_model = is_lvg_model
        self.n_clip_max = n_clip_max

    def _input_shape_checker(self):
        # NOTE: This function must be modified if the model input resolution
        # constraints are changed.
        constraint_shapes = [(576, 1024)]

        assert (self.height, self.width) in constraint_shapes, (
            f"Input shape ({self.height}, {self.width}) is not in supported shapes: "
            f"{list(constraint_shapes)}"
        )

        log.info(
            f"Verified input height and width for compilation: ({self.height}, {self.width})"
        )

    def _init_model(self, rbln_config=None):
        super()._init_model(rbln_config=rbln_config, is_multiview=True)

    def _load_model(self, rbln_config=None):
        super()._load_model(rbln_config=rbln_config, is_multiview=True)

    def _init_network(self, rbln_config=None):
        super()._init_network(rbln_config=rbln_config, is_multiview=True)

    def _load_network(self, rbln_config=None):
        super()._load_network(rbln_config=rbln_config, is_multiview=True)

    def _run_tokenizer_decoding(self, sample: torch.Tensor):
        """Decode latent samples to video frames using the tokenizer decoder.

        Args:
            sample: Latent tensor from diffusion model [B, C, T, H, W]

        Returns:
            np.ndarray: Decoded video frames as uint8 numpy array [T, H, W, C]
                        with values in range [0, 255]
        """

        if self.model.n_views == 5:
            video_arrangement = [1, 0, 2, 3, 0, 4]
        elif self.model.n_views == 6:
            video_arrangement = [1, 0, 2, 4, 3, 5]
        else:
            raise ValueError(f"Unsupported number of views: {self.model.n_views}")
        # Decode video
        video = (1.0 + self.model.decode(sample)).clamp(0, 2) / 2  # [B, 3, T, H, W]
        video_segments = einops.rearrange(
            video, "b c (v t) h w -> b c v t h w", v=self.model.n_views
        )
        grid_video = torch.stack(
            [video_segments[:, :, i] for i in video_arrangement],
            dim=2,
        )
        grid_video = einops.rearrange(
            grid_video, "b c (h w) t h1 w1 -> b c t (h h1) (w w1)", h=2, w=3
        )
        grid_video = (
            (grid_video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()
        )
        video = (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()

        return [grid_video, video]

    def _run_model(
        self,
        embedding: torch.Tensor,
        view_condition_video="",
        initial_condition_video="",
        control_inputs: dict = None,
    ) -> torch.Tensor:
        """Generate video frames using the diffusion model.

        Args:
            prompt_embedding: Text embedding tensor from T5 encoder
            view_condition_video: Path to input sv view condition video
            initial_condition_video: Path to input mv initial frames
            control_inputs: Dictionary of control modalities and corresponding inputs

        Returns:
            Tensor of generated video frames

        Note:
            Model and tokenizer are automatically offloaded after inference
            if offloading is enabled.
        """
        # Get video batch and state shape
        assert len(embedding) == self.model.n_views

        view_condition_video, fps = read_video_or_image_into_frames_BCTHW(
            view_condition_video,
            normalize=False,  # s.t. output range is [0, 255]
            max_frames=6000,
            also_return_fps=True,
        )
        view_condition_video = resize_video(
            view_condition_video,
            self.height,
            self.width,
            interpolation=cv2.INTER_LINEAR,
        )
        view_condition_video = torch.from_numpy(view_condition_video)
        total_T = view_condition_video.shape[2]

        data_batch, state_shape = get_video_batch_for_multiview_model(
            model=self.model,
            prompt_embedding=embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames * len(embedding),
            frame_repeat_negative_condition=0,
        )

        self.model.condition_location = (
            "first_cam_and_first_n" if self.is_lvg_model else "first_cam"
        )

        if self.is_lvg_model:
            if os.path.isdir(initial_condition_video):
                initial_condition_videos = []
                fnames = sorted(os.listdir(initial_condition_video))
                for fname in fnames:
                    if fname.endswith(".mp4"):
                        try:
                            int(fname.split(".")[0])
                        except ValueError:
                            log.warning(
                                f"Could not parse video file name {fname} into view id"
                            )
                            continue
                        initial_condition_video_n = (
                            read_video_or_image_into_frames_BCTHW(
                                fname,
                                normalize=False,
                                max_frames=self.num_input_frames,
                                also_return_fps=True,
                            )
                        )
                        initial_condition_videos.append(
                            torch.from_numpy(initial_condition_video_n)
                        )
                initial_condition_video = torch.cat(initial_condition_videos, dim=2)
            else:
                initial_condition_video, _ = read_video_or_image_into_frames_BCTHW(
                    initial_condition_video,
                    normalize=False,
                    max_frames=6000,
                    also_return_fps=True,
                )  # B C (V T) H W
                initial_condition_video = torch.from_numpy(initial_condition_video)
        else:
            initial_condition_video = None

        data_batch = get_ctrl_batch_mv(
            self.height,
            self.width,
            data_batch,
            total_T,
            control_inputs,
            self.model.n_views,
            self.num_video_frames,
        )  # multicontrol inputs are concatenated channel wise, [-1,1] range

        hint_key = data_batch["hint_key"]
        input_video = None
        control_input = data_batch[hint_key]
        control_weight = data_batch["control_weight"]

        num_new_generated_frames = (
            self.num_video_frames - self.num_input_frames
        )  # 57 - 9 = 48
        B, C, T, H, W = control_input.shape
        T = T // self.model.n_views
        assert T == total_T
        # Different from other examples, we use a different logic to determine
        # total generated duration: we check for the maximum number of clips that
        # can be fit in to the duration of ctrl input and condition input and
        # implicitly trim these videos to that duration
        if self.is_lvg_model:
            N_clip = (T - self.num_input_frames) // num_new_generated_frames
            if self.n_clip_max > 0:
                N_clip = min(self.n_clip_max, N_clip)
        else:
            N_clip = 1
            log.info(
                "Model is not Long-video generation model, overwriting N_clip to 1"
            )

        video = []
        for i_clip in tqdm(range(N_clip)):
            data_batch_i = {k: v for k, v in data_batch.items()}
            start_frame = num_new_generated_frames * i_clip
            end_frame = num_new_generated_frames * (i_clip + 1) + self.num_input_frames

            if input_video is not None:
                x_sigma_max = []
                for b in range(B):
                    input_frames = input_video[b : b + 1, :, start_frame:end_frame]
                    x0 = self.model.encode(input_frames).contiguous()
                    x_sigma_max.append(
                        self.model.get_x_from_clean(
                            x0, self.sigma_max, seed=(self.seed + i_clip)
                        )
                    )
                x_sigma_max = torch.cat(x_sigma_max)
            else:
                x_sigma_max = None

            control_input_BVCT = einops.rearrange(
                control_input, "B C (V T) H W -> (B V) C T H W", V=self.model.n_views
            )
            control_input_i = control_input_BVCT[:, :, start_frame:end_frame]

            data_batch_i[hint_key] = einops.rearrange(
                control_input_i, "(B V) C T H W -> B C (V T) H W", V=self.model.n_views
            )

            condition_input_i = view_condition_video[:, :, start_frame:end_frame]

            latent_hint = []
            for b in range(B):
                data_batch_p = {k: v for k, v in data_batch_i.items()}
                data_batch_p[hint_key] = data_batch_i[hint_key][b : b + 1]
                if len(control_inputs) > 1:
                    latent_hint_i = []
                    for idx in range(0, data_batch_p[hint_key].size(1), 3):
                        x_rgb = data_batch_p[hint_key][:, idx : idx + 3]
                        latent_hint_i.append(self.model.encode(x_rgb))
                    latent_hint.append(torch.cat(latent_hint_i).unsqueeze(0))
                else:
                    latent_hint.append(self.model.encode_latent(data_batch_p))
            data_batch_i["latent_hint"] = latent_hint = torch.cat(latent_hint)

            if isinstance(control_weight, torch.Tensor) and control_weight.ndim > 4:
                control_weight_t = control_weight[..., start_frame:end_frame, :, :]
                t, h, w = latent_hint.shape[-3:]
                data_batch_i["control_weight"] = resize_control_weight_map(
                    control_weight_t, (t, h // 2, w // 2)
                )

            if i_clip == 0:
                if initial_condition_video is not None:
                    prev_frames_blank = torch.zeros(
                        (B, self.model.n_views, C, self.num_video_frames, H, W)
                    ).to(view_condition_video)

                    initial_condition_video_frames_BVCT = einops.rearrange(
                        initial_condition_video,
                        "B C (V T) H W -> B V C T H W",
                        V=self.model.n_views,
                    )
                    prev_frames_blank[:, :, :, : self.num_input_frames] = (
                        initial_condition_video_frames_BVCT[
                            :, :, :, start_frame : start_frame + self.num_input_frames
                        ]
                    )
                    prev_frames = einops.rearrange(
                        prev_frames_blank, "B V C T H W -> B C (V T) H W"
                    )
                    num_input_frames = self.num_input_frames
                else:
                    num_input_frames = 0
                    prev_frames = None
            else:
                num_input_frames = self.num_input_frames
            condition_latent = self.get_condition_latent(
                state_shape,
                data_batch_i,
                cond_video=condition_input_i,
                prev_frames=prev_frames,
                patch_h=H,
                patch_w=W,
                skip_reencode=False,
            ).float()
            # Generate video frames
            latents = generate_world_from_control(
                model=self.model,
                state_shape=self.model.state_shape,
                is_negative_prompt=False,
                data_batch=data_batch_i,
                guidance=self.guidance,
                num_steps=self.num_steps,
                seed=(self.seed + i_clip),
                condition_latent=condition_latent,
                num_input_frames=num_input_frames,
                sigma_max=self.sigma_max if x_sigma_max is not None else None,
                x_sigma_max=x_sigma_max,
                augment_sigma=0.0,
            )
            _, frames = self._run_tokenizer_decoding(latents)  # T H W C
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2)[None]  # 1 C (V T) H W
            frames_BVCT = einops.rearrange(
                frames, "B C (V T) H W -> B V C T H W", V=self.model.n_views
            )
            if i_clip == 0:
                video.append(frames_BVCT)
            else:
                frames_BVCT_non_overlap = frames_BVCT[:, :, :, num_input_frames:]
                video.append(frames_BVCT_non_overlap)

            prev_frames = torch.zeros_like(frames_BVCT)
            n_copy = max(1, abs(self.num_input_frames))
            prev_frames[:, :, :, :n_copy] = frames_BVCT[:, :, :, -n_copy:]
            prev_frames = einops.rearrange(prev_frames, "B V C T H W -> B C (V T) H W")

        video = torch.cat(video, dim=3)
        video = einops.rearrange(video, "B V C T H W -> B C (V T) H W")
        video = video[0].permute(1, 2, 3, 0).numpy()  # T H W C
        return video

    def get_condition_latent(
        self,
        state_shape,
        data_batch_i,
        cond_video=None,
        prev_frames=None,
        patch_h=1024,
        patch_w=1024,
        skip_reencode=False,
        prev_latents=None,
    ):
        """
        Create the condition latent used in this loop for generation from RGB frames
        Args:
            model:
            state_shape: tuple (C T H W), shape of latent to be generated
            data_batch_i: (dict) this is only used to get batch size
            multi_cam: (bool) whether to use multicam processing or revert to
                original behavior from tpsp_demo
            cond_video: (tensor) the front view video for conditioning sv2mv
            prev_frames: (tensor) frames generated in previous loop
            patch_h: (int)
            patch_w: (int)
            skip_reencode: (bool) whether to use the tokenizer to encode
                prev_frames, or read from prev_latents directly
            prev_latents: (tensor) latent generated in previous loop, must not be
                None if skip_reencode

        Returns:

        """
        # this might be not 1 when patching is used
        B = data_batch_i["video"].shape[0]

        latent_sample = (
            torch.zeros(state_shape).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        )  # B, C, (V T), H, W
        latent_sample = einops.rearrange(
            latent_sample, "B C (V T) H W -> B V C T H W", V=self.model.n_views
        )
        log.info(f"model.sigma_data {self.model.sigma_data}")
        if self.model.config.conditioner.video_cond_bool.condition_location.endswith(
            "first_n"
        ):
            if skip_reencode:
                assert prev_latents is not None
                prev_latents = einops.rearrange(
                    prev_latents, "B C (V T) H W -> B V C T H W", V=self.model.n_views
                )
                latent_sample = prev_latents.clone()
            else:
                prev_frames = split_video_into_patches(prev_frames, patch_h, patch_w)
                for b in range(prev_frames.shape[0]):
                    input_frames = prev_frames[b : b + 1] / 255.0 * 2 - 1
                    input_frames = einops.rearrange(
                        input_frames, "1 C (V T) H W -> V C T H W", V=self.model.n_views
                    )
                    encoded_frames = (
                        self.model.tokenizer.encode(input_frames).contiguous()
                        * self.model.sigma_data
                    )
                    latent_sample[b : b + 1, :] = encoded_frames

        if self.model.config.conditioner.video_cond_bool.condition_location.startswith(
            "first_cam"
        ):
            assert cond_video is not None
            cond_video = split_video_into_patches(cond_video, patch_h, patch_w)
            for b in range(cond_video.shape[0]):
                input_frames = cond_video[b : b + 1] / 255.0 * 2 - 1
                # input_frames = einops.rearrange(input_frames, "1 C (V T) H W -> V C T H W",
                #                                  V=self.model.n_views)[:1]
                latent_sample[
                    b : b + 1,
                    0,
                ] = (
                    self.model.tokenizer.encode(input_frames).contiguous()
                    * self.model.sigma_data
                )

        latent_sample = einops.rearrange(latent_sample, " B V C T H W -> B C (V T) H W")
        log.info(f"latent_sample, {latent_sample[:, 0, :, 0, 0]}")

        return latent_sample

    def build_mv_prompt(self, mv_prompts, n_views):
        """
        Apply multiview prompt formatting to the input prompt such that the text
        conditioning matches that used during training.
        Args:
            prompt: caption of one scene, with prompt of each view separated by ";"
            n_views: number of cameras to format the caption to

        Returns:

        """
        if n_views == 5:
            base_prompts = [
                "The video is captured from a camera mounted on a car. "
                "The camera is facing forward.",
                "The video is captured from a camera mounted on a car. "
                "The camera is facing to the left.",
                "The video is captured from a camera mounted on a car. "
                "The camera is facing to the right.",
                "The video is captured from a camera mounted on a car. "
                "The camera is facing the rear left side.",
                "The video is captured from a camera mounted on a car. "
                "The camera is facing the rear right side.",
            ]
        elif n_views == 6:
            base_prompts = [
                "The video is captured from a camera mounted on a car. "
                "The camera is facing forward.",
                "The video is captured from a camera mounted on a car. "
                "The camera is facing to the left.",
                "The video is captured from a camera mounted on a car. "
                "The camera is facing to the right.",
                "The video is captured from a camera mounted on a car. "
                "The camera is facing backwards.",
                "The video is captured from a camera mounted on a car. "
                "The camera is facing the rear left side.",
                "The video is captured from a camera mounted on a car. "
                "The camera is facing the rear right side.",
            ]

        log.info(f"Reading multiview prompts, found {len(mv_prompts)} splits")
        n = len(mv_prompts)
        if n < n_views:
            mv_prompts += base_prompts[n:]
        else:
            mv_prompts = mv_prompts[:n_views]

        for vid, p in enumerate(mv_prompts):
            if not p.startswith(base_prompts[vid]):
                mv_prompts[vid] = base_prompts[vid] + " " + p
                log.info(f"Adding missing camera caption to view {vid}, {p[:30]}")

        log.info(f"Procced multiview prompts, {len(mv_prompts)} splits")
        return mv_prompts

    def generate(
        self,
        prompts: list,
        view_condition_video: str,
        initial_condition_video: str,
        control_inputs: dict = None,
        save_folder: str = "outputs/",
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt and control video.

        Pipeline steps:
        1. Run safety checks on input prompt
        2. Convert prompt to embeddings
        3. Generate video frames using diffusion
        4. Run safety checks and apply face blur on generated video frames

        Args:
            control_inputs: Control inputs for guided generation
            save_folder: Folder to save intermediate files

        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """

        log.info(f"Run with view condition video path: {view_condition_video}")
        if initial_condition_video:
            log.info(
                f"Run with initial condition video path: {initial_condition_video}"
            )
        mv_prompts = self.build_mv_prompt(prompts, self.model.n_views)
        log.info(f"Run with prompt: {mv_prompts}")

        # Process prompts into multiview format
        log.info("Run guardrail on prompt")
        is_safe = self._run_guardrail_on_prompt(". ".join(mv_prompts))
        if not is_safe:
            log.critical("Input text prompt is not safe")
            return None
        log.info("Pass guardrail on prompt")

        prompt_embeddings, _ = self._run_text_embedding_on_prompt(mv_prompts)
        prompt_embedding = torch.concat(prompt_embeddings, dim=0)

        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")

        video = self._run_model(
            prompt_embedding,
            view_condition_video,
            initial_condition_video,
            control_inputs=control_inputs,
        )
        log.info("Finish generation")
        log.info("Run guardrail on generated video")
        video = self._run_guardrail_on_video(video)
        if video is None:
            log.critical("Generated video is not safe")
            raise ValueError("Guardrail check failed: Generated video is unsafe")

        log.info("Pass guardrail on generated video")

        return video, mv_prompts


class RBLNDistilledControl2WorldGenerationPipeline(
    RBLNDiffusionControl2WorldGenerationPipeline
):
    """Pipeline for distilled ControlNet video2video inference."""

    def _init_model(self, rbln_config=None):
        log.info("Compiling distilled base model")
        log.info("Loading distilled consolidated checkpoint")
        model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos-transfer1/cosmos_transfer1/diffusion/config/transfer/config.py",
            model_class=self.model_class,
            base_checkpoint_dir=self.checkpoint_dir,
        )

        model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
        with skip_init_linear():
            model.set_up_model()

        assert len(self.control_inputs) == 1, (
            "Distilled model only supports single control input"
        )

        for _, config in self.control_inputs.items():
            checkpoint_path = config["ckpt_path"]

        log.info(f"Loading base model and ctrl model from ckpt_path: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)

        # Split into base and control components
        base_state_dict = {}
        ctrl_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith("net.base_model.net."):
                base_key = k[len("net.base_model.net.") :]
                base_state_dict[base_key] = v
            elif k.startswith("net.net_ctrl."):
                ctrl_key = k[len("net.net_ctrl.") :]
                ctrl_state_dict[ctrl_key] = v

        # Load base model weights
        if base_state_dict:
            model.model.net.base_model.net.load_state_dict(
                base_state_dict, strict=False
            )
        # Load control weights
        if ctrl_state_dict:
            model.model.net.net_ctrl.load_state_dict(ctrl_state_dict, strict=False)

        hidden_size = model.model.net.base_model.net.model_channels
        attention_head_dim = hidden_size // model.model.net.base_model.net.num_heads
        spatial_compression_factor = (
            model.config.tokenizer.video_vae.spatial_compression_factor
        )
        temporal_compression_factor = (
            model.config.tokenizer.video_vae.temporal_compression_factor
        )
        patch_spatial = model.model.net.base_model.net.patch_spatial
        patch_temporal = model.model.net.base_model.net.patch_temporal

        latent_height = self.height // spatial_compression_factor
        latent_width = self.width // spatial_compression_factor
        num_latent_frames = (
            self.num_video_frames - 1
        ) // temporal_compression_factor + 1

        hidden_dim = (
            (latent_height // patch_spatial)
            * (latent_width // patch_spatial)
            * num_latent_frames
            // patch_temporal
        )

        input_info = [
            (
                "x",
                [
                    num_latent_frames,
                    latent_height // patch_spatial,
                    latent_width // patch_spatial,
                    1,
                    hidden_size,
                ],
                torch.float32,
            ),
            ("affline_emb_B_D", [1, hidden_size], torch.float32),
            ("crossattn_emb", [512, 1, 1024], torch.float32),
            ("cos_", [hidden_dim, 1, 1, attention_head_dim], torch.float32),
            ("sin_", [hidden_dim, 1, 1, attention_head_dim], torch.float32),
            ("adaln_lora_B_3D", [1, hidden_size * 3], torch.float32),
            (
                "extra_per_block_pos_emb",
                [
                    num_latent_frames,
                    latent_height // patch_spatial,
                    latent_width // patch_spatial,
                    1,
                    hidden_size,
                ],
                torch.float32,
            ),
        ]

        original_shape = torch.Size([1, 17, 16, latent_height, latent_width])

        model.model.net.base_model.net.to(torch.float32)
        wrapped_model = GeneralDITWrapperWithoutRegion(
            model.model.net.base_model.net, original_shape
        )

        num_ctrl_blocks = len(model.config.net_ctrl.layer_mask) - sum(
            model.config.net_ctrl.layer_mask
        )
        for i in range(num_ctrl_blocks):
            input_info.append(
                (
                    f"x_ctrl_{i}",
                    [
                        num_latent_frames,
                        latent_height // patch_spatial,
                        latent_width // patch_spatial,
                        1,
                        hidden_size,
                    ],
                    torch.float32,
                )
            )

        self.compiled_model = rebel.compile_from_torch(
            wrapped_model,
            input_info,
            num_devices=rbln_config.get("num_devices", None),
        )
        base_model = model.model.net.base_model.net
        runtime = RBLNVideoExtendGeneralDIT(
            self.compiled_model, base_model, rbln_config
        )
        if self.create_runtimes:
            runtime.create_runtime()

        delattr(model.model.net.base_model, "net")
        setattr(model.model.net.base_model, "net", runtime)
        self.model = model

    def _load_model(self, rbln_config=None):
        model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos-transfer1/cosmos_transfer1/diffusion/config/transfer/config.py",
            model_class=self.model_class,
            base_checkpoint_dir=self.checkpoint_dir,
        )

        model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
        with skip_init_linear():
            model.set_up_model()

        assert len(self.control_inputs) == 1, (
            "Distilled model only supports single control input"
        )

        for _, config in self.control_inputs.items():
            checkpoint_path = config["ckpt_path"]

        log.info(f"Loading base model and ctrl model from ckpt_path: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)

        # Split into base and control components
        base_state_dict = {}
        ctrl_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith("net.base_model.net."):
                base_key = k[len("net.base_model.net.") :]
                base_state_dict[base_key] = v
            elif k.startswith("net.net_ctrl."):
                ctrl_key = k[len("net.net_ctrl.") :]
                ctrl_state_dict[ctrl_key] = v

        # Load base model weights
        if base_state_dict:
            model.model.net.base_model.net.load_state_dict(
                base_state_dict, strict=False
            )
        # Load control weights
        if ctrl_state_dict:
            model.model.net.net_ctrl.load_state_dict(ctrl_state_dict, strict=False)

        compiled_model = rebel.RBLNCompiledModel(
            os.path.join(self.rbln_dir, "transformer.rbln")
        )
        base_model = model.model.net.base_model.net
        runtime = RBLNVideoExtendGeneralDIT(compiled_model, base_model, rbln_config)
        if self.create_runtimes:
            runtime.create_runtime()

        delattr(model.model.net.base_model, "net")
        setattr(model.model.net.base_model, "net", runtime)
        self.model = model

    def _init_network(self, rbln_config=None):
        log.info("Compiling distilled ctrl models")
        hint_encoders = {}
        compiled_controlnet = {}

        assert len(self.control_inputs) == 1, (
            "Distilled model only supports single control input"
        )
        key = list(self.control_inputs)[0]

        hidden_size = self.model.model.net.base_model.net.base_model.model_channels
        attention_head_dim = (
            hidden_size // self.model.model.net.base_model.net.base_model.num_heads
        )
        spatial_compression_factor = (
            self.model.config.tokenizer.video_vae.spatial_compression_factor
        )
        temporal_compression_factor = (
            self.model.config.tokenizer.video_vae.temporal_compression_factor
        )
        patch_spatial = self.model.model.net.base_model.net.base_model.patch_spatial
        patch_temporal = self.model.model.net.base_model.net.base_model.patch_temporal

        latent_height = self.height // spatial_compression_factor
        latent_width = self.width // spatial_compression_factor
        num_latent_frames = (
            self.num_video_frames - 1
        ) // temporal_compression_factor + 1

        hidden_dim = (
            (latent_height // patch_spatial)
            * (latent_width // patch_spatial)
            * num_latent_frames
            // patch_temporal
        )
        input_info = [
            (
                "guided_hints",
                [
                    num_latent_frames,
                    latent_height // patch_spatial,
                    latent_width // patch_spatial,
                    1,
                    hidden_size,
                ],
                torch.float32,
            ),
            (
                "x",
                [
                    num_latent_frames,
                    latent_height // patch_spatial,
                    latent_width // patch_spatial,
                    1,
                    hidden_size,
                ],
                torch.float32,
            ),
            ("cos_", [hidden_dim, 1, 1, attention_head_dim], torch.float32),
            ("sin_", [hidden_dim, 1, 1, attention_head_dim], torch.float32),
            (
                "extra_pos_emb",
                [
                    num_latent_frames,
                    latent_height // patch_spatial,
                    latent_width // patch_spatial,
                    1,
                    hidden_size,
                ],
                torch.float32,
            ),
            ("crossattn_emb", [512, 1, 1024], torch.float32),
            ("adaln_lora_B_3D", [1, hidden_size * 3], torch.float32),
            ("affline_emb_B_D", [1, hidden_size], torch.float32),
            ("control_weight", [1], torch.float32),
        ]
        self.model.model.net.net_ctrl.to(torch.float32)
        wrapped_controlnet = ControlNetWrapper(self.model.model.net.net_ctrl).eval()
        compiled_model = rebel.compile_from_torch(
            wrapped_controlnet,
            input_info,
            num_devices=rbln_config.get(key, {}).get("num_devices", None),
        )

        compiled_controlnet[key] = compiled_model
        controlnet_runtime = RBLNRuntimeControlNet(
            compiled_model, self.model.model.net.net_ctrl, rbln_config.get(key, {})
        )
        if self.create_runtimes:
            controlnet_runtime.create_runtime()

        hint_encoders[key] = controlnet_runtime

        # generaldit attrs
        in_channels = self.model.model.net.net_ctrl.in_channels
        use_cross_attn_mask = self.model.model.net.net_ctrl.use_cross_attn_mask
        self.compiled_controlnet = compiled_controlnet

        delattr(self.model.model.net, "net_ctrl")
        setattr(
            self.model.model.net,
            "net_ctrl",
            RBLNGeneralDITEncoder(
                hint_encoders,
                self.control_inputs.keys(),
                in_channels,
                use_cross_attn_mask,
            ),
        )

    def _load_network(self, rbln_config):
        hint_encoders = {}
        compiled_controlnet = {}

        assert len(self.control_inputs) == 1, (
            "Distilled model only supports single control input"
        )
        key = list(self.control_inputs)[0]

        compiled_controlnet[key] = rebel.RBLNCompiledModel(
            os.path.join(self.rbln_dir, "ctrlnet", f"{key}.rbln")
        )
        controlnet_runtime = RBLNRuntimeControlNet(
            compiled_controlnet[key],
            self.model.model.net.net_ctrl,
            rbln_config.get(key, {}),
        )
        if self.create_runtimes:
            controlnet_runtime.create_runtime()

        hint_encoders[key] = controlnet_runtime

        # generaldit attrs
        in_channels = self.model.model.net.net_ctrl.in_channels
        use_cross_attn_mask = self.model.model.net.net_ctrl.use_cross_attn_mask

        self.compiled_controlnet = compiled_controlnet

        delattr(self.model.model.net, "net_ctrl")
        setattr(
            self.model.model.net,
            "net_ctrl",
            RBLNGeneralDITEncoder(
                hint_encoders,
                self.control_inputs.keys(),
                in_channels,
                use_cross_attn_mask,
            ),
        )

    def _run_model(
        self,
        prompt_embeddings: torch.Tensor,  # [B, ...]
        video_paths: list[str],  # [B]
        negative_prompt_embeddings: Optional[torch.Tensor] = None,  # [B, ...] or None
        control_inputs_list: list[dict] = None,  # [B] list of dicts
    ) -> np.ndarray:
        """
        Batched world generation with model offloading.
        Each batch element corresponds to a (prompt, video, control_inputs) triple.
        """
        prompt_embeddings = torch.cat(prompt_embeddings)
        if negative_prompt_embeddings is not None:
            negative_prompt_embeddings = torch.cat(negative_prompt_embeddings)

        B = len(video_paths)
        assert len(prompt_embeddings) == B, "Batch size mismatch for prompt embeddings"
        if negative_prompt_embeddings is not None:
            assert len(negative_prompt_embeddings) == B, (
                "Batch size mismatch for negative prompt embeddings"
            )
        assert len(control_inputs_list) == B, (
            "Batch size mismatch for control_inputs_list"
        )

        log.info("Starting data augmentation")

        log.info(
            "Regional prompts not supported when using distilled model, "
            f"dropping: {self.regional_prompts}"
        )

        # Get video batch and state shape
        data_batch, state_shape = get_batched_ctrl_batch(
            model=self.model,
            prompt_embeddings=prompt_embeddings,  # [B, ...]
            negative_prompt_embeddings=negative_prompt_embeddings,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames,
            input_video_paths=video_paths,  # [B]
            control_inputs_list=control_inputs_list,  # [B]
            blur_strength=self.blur_strength,
            canny_threshold=self.canny_threshold,
        )

        if data_batch["input_video"] is None:
            ctrl_video = next(
                (v for k, v in data_batch.items() if k.startswith("control_input")),
                None,
            )
            height, width = ctrl_video.shape[3], ctrl_video.shape[4]
        else:
            input_video = data_batch["input_video"]
            height, width = input_video.shape[3], input_video.shape[4]

        assert height == self.height, (
            f"Resized input video height {height} does "
            f"not match compiled model input height {self.height}, You must "
            f"re-compile the model with the new height that is the same as the "
            f"resized input video height."
        )
        assert width == self.width, (
            f"Resized input video width {width} does "
            f"not match compiled model input width {self.width}, You must "
            f"re-compile the model with the new width that is the same as the "
            f"resized input video width."
        )

        log.info("Completed data augmentation")

        hint_key = data_batch["hint_key"]
        control_input = data_batch[hint_key]  # [B, C, T, H, W]
        input_video = data_batch.get("input_video", None)
        control_weight = data_batch.get("control_weight", None)
        num_new_generated_frames = self.num_video_frames - self.num_input_frames
        B, C, T, H, W = control_input.shape
        if (
            T - self.num_input_frames
        ) % num_new_generated_frames != 0:  # pad duplicate frames at the end
            pad_t = num_new_generated_frames - (
                (T - self.num_input_frames) % num_new_generated_frames
            )
            pad_frames = control_input[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
            control_input = torch.cat([control_input, pad_frames], dim=2)
            if input_video is not None:
                pad_video = input_video[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
                input_video = torch.cat([input_video, pad_video], dim=2)
            num_total_frames_with_padding = control_input.shape[2]
            if (
                isinstance(control_weight, torch.Tensor)
                and control_weight.ndim > 5
                and num_total_frames_with_padding > control_weight.shape[3]
            ):
                pad_t = num_total_frames_with_padding - control_weight.shape[3]
                pad_weight = control_weight[:, :, :, -1:].repeat(1, 1, 1, pad_t, 1, 1)
                control_weight = torch.cat([control_weight, pad_weight], dim=3)
        else:
            num_total_frames_with_padding = T
        N_clip = (
            num_total_frames_with_padding - self.num_input_frames
        ) // num_new_generated_frames

        video = []
        prev_frames = None
        log.info(f"N_clip: {N_clip}")
        for i_clip in tqdm(range(N_clip)):
            data_batch_i = {k: v for k, v in data_batch.items()}
            start_frame = num_new_generated_frames * i_clip
            end_frame = num_new_generated_frames * (i_clip + 1) + self.num_input_frames

            data_batch_i[hint_key] = control_input[:, :, start_frame:end_frame]
            latent_hint = []
            log.info("Starting latent encoding")
            for b in range(B):
                data_batch_p = {k: v for k, v in data_batch_i.items()}
                data_batch_p[hint_key] = data_batch_i[hint_key][b : b + 1]
                if len(control_inputs_list) >= 1 and len(control_inputs_list[0]) > 1:
                    latent_hint_i = []
                    for idx in range(0, data_batch_p[hint_key].size(1), 3):
                        x_rgb = data_batch_p[hint_key][:, idx : idx + 3]
                        latent_hint_i.append(self.model.encode(x_rgb))
                    latent_hint.append(torch.cat(latent_hint_i).unsqueeze(0))
                else:
                    for k, v in data_batch_p.items():
                        if isinstance(v, torch.Tensor):
                            if v.dtype == torch.bfloat16:
                                data_batch_p[k] = v.to(torch.float32)
                    latent_hint.append(self.model.encode_latent(data_batch_p))
            data_batch_i["latent_hint"] = latent_hint = torch.cat(latent_hint)
            log.info("Completed latent encoding")

            # Resize control_weight if needed
            if isinstance(control_weight, torch.Tensor) and control_weight.ndim > 4:
                control_weight_t = control_weight[..., start_frame:end_frame, :, :]
                t, h, w = latent_hint.shape[-3:]
                data_batch_i["control_weight"] = resize_control_weight_map(
                    control_weight_t, (t, h // 2, w // 2)
                )

            if i_clip == 0:
                num_input_frames = 0
                latent_tmp = latent_hint if latent_hint.ndim == 5 else latent_hint[:, 0]
                condition_latent = torch.zeros_like(latent_tmp)
            else:
                num_input_frames = self.num_input_frames
                prev_frames = split_video_into_patches(
                    prev_frames, control_input.shape[-2], control_input.shape[-1]
                )
                input_frames = prev_frames.float32() / 255.0 * 2 - 1
                condition_latent = self.model.encode(input_frames).contiguous()

            # Generate video frames for this clip (batched)
            log.info("Starting diffusion sampling")
            latents = generate_world_from_control(
                model=self.model,
                state_shape=state_shape,
                is_negative_prompt=False,  # Unused for distilled models
                data_batch=data_batch_i,
                guidance=self.guidance,
                num_steps=self.num_steps,
                seed=(self.seed + i_clip),
                condition_latent=condition_latent,
                num_input_frames=num_input_frames,
                sigma_max=None,
            )
            log.info("Completed diffusion sampling")

            log.info("Starting VAE decode")
            frames = self._run_tokenizer_decoding(latents)  # [B, T, H, W, C] or similar
            log.info("Completed VAE decode")

            if i_clip == 0:
                video.append(frames)
            else:
                video.append(frames[:, :, self.num_input_frames :])

            prev_frames = torch.zeros_like(frames)
            prev_frames[:, :, : self.num_input_frames] = frames[
                :, :, -self.num_input_frames :
            ]

        video = torch.cat(video, dim=2)[:, :, :T]
        video = video.permute(0, 2, 3, 4, 1).numpy()
        return video
