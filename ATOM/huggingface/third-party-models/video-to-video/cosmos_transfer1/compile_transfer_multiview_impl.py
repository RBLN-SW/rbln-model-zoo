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
    BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH,
    BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH,
)
from cosmos_transfer1.diffusion.inference.inference_utils import (
    default_model_names,
    load_controlnet_specs,
    valid_hint_keys,
)
from cosmos_transfer1.utils import log
from rbln_inference.rbln_pipeline import (
    RBLNDiffusionControl2WorldMultiviewGenerationPipeline,
)

torch.serialization.add_safe_globals([BytesIO])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control to world generation demo script",
        conflict_handler="resolve",
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
    parser.add_argument("--is_lvg_model", action="store_true", help="")
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
        "--use_perf", action="store_true", help="using performance mode"
    )
    cmd_args = parser.parse_args()

    # Load and parse JSON input
    control_inputs, json_args = load_controlnet_specs(cmd_args)
    for key in json_args:
        if f"--{key}" not in sys.argv:
            setattr(cmd_args, key, json_args[key])

    return cmd_args, control_inputs


def validate_controlnet_specs(cfg, controlnet_specs):
    """
    Load and validate controlnet specifications from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing controlnet specs.
        checkpoint_dir (str): Base directory for checkpoint files.

    Returns:
        Dict[str, Any]: Validated and processed controlnet specifications.
    """
    checkpoint_dir = cfg.checkpoint_dir

    for hint_key, config in controlnet_specs.items():
        if hint_key not in list(valid_hint_keys) + ["prompts", "view_condition_video"]:
            raise ValueError(
                f"Invalid hint_key: {hint_key}. Must be one of {valid_hint_keys}"
            )
        if hint_key in valid_hint_keys:
            if "ckpt_path" not in config:
                log.info(f"No checkpoint path specified for {hint_key}. Using default.")
                config["ckpt_path"] = os.path.join(
                    checkpoint_dir, default_model_names[hint_key]
                )
            elif not os.path.exists(config["ckpt_path"]):
                ckpt_path = os.path.join(checkpoint_dir, config["ckpt_path"])
                config["ckpt_path"] = ckpt_path

            # Regardless whether "control_weight_prompt" is provided (i.e. whether we automatically
            # generate spatiotemporal control weight binary masks), control_weight is needed to.
            if "control_weight" not in config:
                log.warning(
                    f"No control weight specified for {hint_key}. Setting to 0.5."
                )
                config["control_weight"] = "0.5"
            else:
                # Check if control weight is a path or a scalar
                weight = config["control_weight"]
                if not isinstance(weight, str) or not weight.endswith(".pt"):
                    try:
                        # Try converting to float
                        scalar_value = float(weight)
                        if scalar_value < 0:
                            raise ValueError(
                                f"Control weight for {hint_key} must be non-negative."
                            )
                    except ValueError:
                        raise ValueError(
                            f"Control weight for {hint_key} must be a valid non-negative float "
                            f"or a path to a .pt file."
                        )

    return controlnet_specs


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

    rbln_config = {
        "transformer": {"num_devices": 4},
        "safety_checker": {"qwen3guard": {"num_devices": 1}},
    }

    if cfg.use_perf:
        rbln_config["transformer"].update({"num_devices": 8})

    ctrlnet_rbln_config = {}
    for ctrl in control_inputs.keys():
        ctrlnet_rbln_config[ctrl] = {"num_devices": 2}

    rbln_config["ctrlnet"] = ctrlnet_rbln_config

    if cfg.is_lvg_model:
        checkpoint = BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH
    else:
        checkpoint = BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH

    for key in rbln_config["ctrlnet"].keys():
        rbln_config["ctrlnet"][key].update({"num_devices": 2})

    pipeline = RBLNDiffusionControl2WorldMultiviewGenerationPipeline(
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_name=checkpoint,
        batch_size=cfg.batch_size,
        control_inputs=control_inputs,
        num_video_frames=57,
        height=576,
        width=1024,
        is_lvg_model=cfg.is_lvg_model,
        rbln_config=rbln_config,
        export=True,
        create_runtimes=False,
        disable_guardrail=False,
    )

    pipeline.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
