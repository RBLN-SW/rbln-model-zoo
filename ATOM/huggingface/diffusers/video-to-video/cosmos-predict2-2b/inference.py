import argparse
import os

import torch
from diffusers.utils import export_to_video, load_image
from optimum.rbln import RBLNCosmos2VideoToWorldPipeline

DEFAULT_PROMPT = (
    "A high-definition video captures the precision of robotic welding in an industrial setting. The first frame "
    "showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding "
    "process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display "
    "of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the "
    "view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including "
    "a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As "
    "the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving "
    "to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing "
    "slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. "
    "The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, "
    "underscoring the ongoing nature of the welding operation."
)


def parsing_argument():
    parser = argparse.ArgumentParser(description="Run Cosmos Predict2 Video2World 2B")
    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_PROMPT,
        help="(str) type, Text prompt for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="(int) type, Random seed for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "nvidia/Cosmos-Predict2-2B-Video2World"

    # To condition on a video instead, cut the clip yourself. This pipeline has no
    # `num_conditional_frames` argument and uses the full input as conditioning, so a 93-frame clip
    # pins all 24 latent frames and the model only reproduces the input. Give it the last 5 frames,
    # which is the reference implementation's multi-frame setting (`--num_conditional_frames 5`):
    #     video = load_video("input.mp4")[-5:]
    #     video = pipe(video=video, prompt=args.text, ...).frames[0]
    image = load_image(
        "https://media.githubusercontent.com/media/nvidia-cosmos/cosmos-predict2.5/refs/heads/main/assets/base/robot_welding.jpg"
    )

    # Load all pipeline compiled model
    pipe = RBLNCosmos2VideoToWorldPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        rbln_config={
            # The `rbln_config` is a dictionary used to pass configurations for the model and its submodules.
            # The `device` parameter specifies which device should be used for each submodule during runtime.
            #
            # Since Cosmos-Predict2 Video2World consists of multiple submodules, loading them all onto a
            # single device may exceed its memory capacity. Therefore, when creating runtimes for
            # each submodule, devices can be divided and assigned for efficient memory utilization.
            #
            # For example — approximate per-submodule memory footprints (rebel-compiler 0.11.3),
            # used to plan the device assignment below. Assume ~15.7 GiB per device (e.g. RBLN-CA25):
            #   text_encoder     ~2.4 GiB per device (4 devices, T5)
            #   transformer      ~3.0 GiB on device 0, ~1.8 GiB on the others (8 devices)
            #   VAE encoder      ~9.2 GiB
            #   VAE decoder      ~11.4 GiB
            #   safety_checker   qwen3guard ~1.3 GiB, face_blur_filter ~130 MiB
            #
            # As of cosmos-guardrail 0.3.1 the video guardrail keeps only `face_blur_filter`;
            # the SigLIP content-safety filter is disabled upstream, so `siglip_encoder`
            # and `video_safety_model` are no longer compiled.
            "text_encoder": {
                "device": [0, 1, 2, 3],
            },
            "transformer": {
                "device": [0, 1, 2, 3, 4, 5, 6, 7],
            },
            "vae": {
                "device_map": {"encoder": 0, "decoder": 4},
            },
            "safety_checker": {
                "qwen3guard": {"device": 1},
                "face_blur_filter": {"device": 1},
            },
        },
    )

    print(f"Input prompt: {args.text}")

    # Inference with image & text pair and Generate video from them
    video = pipe(
        image=image,
        prompt=args.text,
        generator=torch.Generator().manual_seed(args.seed),
    ).frames[0]
    export_to_video(video, f"{os.path.basename(model_id)}.mp4", fps=16)


if __name__ == "__main__":
    main()
