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

import argparse
import os
import sys
from io import BytesIO

import set_environments  # noqa: F401
import torch
from cosmos_transfer1.checkpoints import (
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH,
    BASE_7B_CHECKPOINT_PATH,
    EDGE2WORLD_CONTROLNET_7B_DISTILLED_CHECKPOINT_PATH,
)
from cosmos_transfer1.diffusion.inference.inference_utils import (
    load_controlnet_specs,
    validate_controlnet_specs,
)
from cosmos_transfer1.utils import log
from rbln_auxiliary import Preprocessors
from rbln_inference.rbln_pipeline import (
    RBLNDiffusionControl2WorldGenerationPipeline,
    RBLNDistilledControl2WorldGenerationPipeline,
)

torch.serialization.add_safe_globals([BytesIO])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control to world generation demo script",
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--is_av_sample",
        action="store_true",
        help="Whether the model is an driving post-training model",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Base directory containing model checkpoints",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="Cosmos-Tokenize1-CV8x8x8-720p",
        help="Tokenizer weights directory relative to checkpoint_dir",
    )
    parser.add_argument(
        "--controlnet_specs",
        type=str,
        help="Path to JSON file specifying multicontrolnet configurations",
        required=True,
    )
    parser.add_argument(
        "--input_video_path",
        type=str,
        default="test.mp4",
        help="Optional input RGB video path",
    )
    parser.add_argument(
        "--rbln_dir",
        type=str,
        required=True,
        help="Base directory containing compiled models",
    )
    parser.add_argument(
        "--sigma_max", type=float, default=80.0, help="sigma_max for partial denoising"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--fps", type=int, default=24, help="FPS of the output video")
    parser.add_argument(
        "--use_distilled",
        action="store_true",
        help="Use distilled ControlNet model variant",
    )
    parser.add_argument(
        "--upsample_prompt",
        action="store_true",
        help="Upsample prompt using Pixtral upsampler model",
    )
    parser.add_argument(
        "--use_regional_prompts",
        action="store_true",
        help="Compile model that uses regional prompts",
    )
    parser.add_argument(
        "--num_regions",
        type=int,
        default=None,
        help="number of regions in regional prompts",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="(int) type, height of the image.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="(int) type, width of the image.",
    )
    parser.add_argument(
        "--use_perf", action="store_true", help="using performance mode"
    )
    cmd_args = parser.parse_args()

    # Load and parse JSON input
    control_inputs, json_args = load_controlnet_specs(cmd_args)
    for key in json_args:
        if f"--{key}" not in sys.argv:
            setattr(cmd_args, key, json_args[key])

    return cmd_args, control_inputs


def main():
    """Run control-to-world generation demo.

    This function handles the main control-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from prompts and images/videos
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts/images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    cfg, control_inputs = parse_arguments()

    if cfg.batch_size != 1:
        cfg.batch_size = 1
        log.critical("Setting batch_size=1 as we only support batch_size=1")

    control_inputs = validate_controlnet_specs(cfg, control_inputs)
    save_dir = cfg.rbln_dir

    if cfg.use_distilled:
        log.info(
            "Regional attention is not supported when using distilled model."
            "Set use_regional_prompts option to False"
        )
        cfg.use_regional_prompts = False

    rbln_config = {
        "transformer": {"num_devices": 4},
        "safety_checker": {"qwen3guard": {"num_devices": 1}},
    }

    if "upscale" in control_inputs.keys():
        rbln_config["safety_checker"]["is_upscale"] = True

    ctrlnet_rbln_config = {}
    if cfg.use_perf:
        rbln_config["transformer"].update({"num_devices": 8})
        for ctrl in control_inputs.keys():
            ctrlnet_rbln_config[ctrl] = {"num_devices": 2}

    rbln_config["ctrlnet"] = ctrlnet_rbln_config

    Preprocessors(
        export=True,
        model_save_dir=os.path.join(save_dir, "preprocessor"),
    )

    # Initialize transfer generation model pipeline
    if cfg.use_distilled:
        checkpoint = EDGE2WORLD_CONTROLNET_7B_DISTILLED_CHECKPOINT_PATH
        pipeline = RBLNDistilledControl2WorldGenerationPipeline(
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoint_name=checkpoint,
            batch_size=cfg.batch_size,
            control_inputs=control_inputs,
            upsample_prompt=cfg.upsample_prompt,
            rbln_config=rbln_config,
            height=cfg.height,
            width=cfg.width,
            export=True,
            create_runtimes=False,
        )
    else:
        checkpoint = (
            BASE_7B_CHECKPOINT_AV_SAMPLE_PATH
            if cfg.is_av_sample
            else BASE_7B_CHECKPOINT_PATH
        )
        pipeline = RBLNDiffusionControl2WorldGenerationPipeline(
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoint_name=checkpoint,
            batch_size=cfg.batch_size,
            control_inputs=control_inputs,
            regional_prompts=cfg.use_regional_prompts,
            num_regions=cfg.num_regions,
            upsample_prompt=cfg.upsample_prompt,
            rbln_config=rbln_config,
            height=cfg.height,
            width=cfg.width,
            export=True,
            create_runtimes=False,
        )

    pipeline.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
