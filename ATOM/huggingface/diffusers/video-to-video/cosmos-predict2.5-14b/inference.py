import argparse
import os

import torch
from diffusers.utils import export_to_video, load_video
from optimum.rbln import RBLNCosmos2_5_PredictBasePipeline

DEFAULT_PROMPT = (
    "A robotic arm, primarily white with black joints and cables, is shown in a clean, modern indoor setting with a "
    "white tabletop. The arm, equipped with a gripper holding a small, light green pitcher, is positioned above a "
    "clear glass containing a reddish-brown liquid and a spoon. The robotic arm is in the process of pouring a "
    "transparent liquid into the glass. To the left of the pitcher, there is an opened jar with a similar "
    "reddish-brown substance visible through its transparent body. In the background, a vase with white flowers and a "
    "brown couch are partially visible, adding to the contemporary ambiance. The lighting is bright, casting soft "
    "shadows on the table. The robotic arm's movements are smooth and controlled, demonstrating precision in its "
    "task. As the video progresses, the robotic arm completes the pour, leaving the glass half-filled with the "
    "reddish-brown liquid. The jar remains untouched throughout the sequence, and the spoon inside the glass remains "
    "stationary. The other robotic arm on the right side also stays stationary throughout the video. The final frame "
    "captures the robotic arm with the pitcher finishing the pour, with the glass now filled to a higher level, while "
    "the pitcher is slightly tilted but still held securely by the gripper."
)


def parsing_argument():
    parser = argparse.ArgumentParser(description="Run Cosmos Predict2.5 14B")
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
    model_id = "nvidia/Cosmos-Predict2.5-14B"

    # Conditioning video: Video2World continues from the end of the input clip. The pipeline
    # itself keeps only the last 5 frames (num_latent_conditional_frames=2).
    input_video = load_video(
        "https://media.githubusercontent.com/media/nvidia-cosmos/cosmos-predict2.5/refs/heads/main/assets/base/robot_pouring.mp4"
    )

    # Load all pipeline compiled model
    pipe = RBLNCosmos2_5_PredictBasePipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        rbln_config={
            # The `rbln_config` is a dictionary used to pass configurations for the model and its submodules.
            # The `device` parameter specifies which device should be used for each submodule during runtime.
            #
            # Since Cosmos-Predict2.5 consists of multiple submodules, loading them all onto a
            # single device may exceed its memory capacity. Therefore, when creating runtimes for
            # each submodule, devices can be divided and assigned for efficient memory utilization.
            #
            # For example — approximate per-submodule memory footprints (rebel-compiler 0.11.3),
            # used to plan the device assignment below. Assume ~15.7 GiB per device (e.g. RBLN-CA25):
            #   text_encoder     ~1.9 GiB per device (8 devices)
            #   transformer      ~9.4 GiB on device 0, ~7.4 GiB on device 4, ~6.6 GiB on the rest
            #                    (8 devices; 40 attention heads split 5 per device)
            #   VAE encoder      ~9.2 GiB
            #   VAE decoder      ~11.4 GiB
            #   safety_checker   qwen3guard ~1.3 GiB, face_blur_filter ~130 MiB
            #
            # As of cosmos-guardrail 0.3.1 the video guardrail keeps only `face_blur_filter`;
            # the SigLIP content-safety filter is disabled upstream, so `siglip_encoder`
            # and `video_safety_model` are no longer compiled.
            "text_encoder": {
                "device": [0, 1, 2, 3, 4, 5, 6, 7],
            },
            "transformer": {
                "device": [0, 1, 2, 3, 4, 5, 6, 7],
            },
            "vae": {
                "device_map": {"encoder": 8, "decoder": 9},
            },
            "safety_checker": {
                "qwen3guard": {"device": 0},
                "face_blur_filter": {"device": 0},
            },
        },
    )

    print(f"Input prompt: {args.text}")

    # Inference with video & text pair and Generate video from them
    video = pipe(
        video=input_video,
        prompt=args.text,
        num_frames=93,
        generator=torch.Generator().manual_seed(args.seed),
    ).frames[0]
    export_to_video(video, f"{os.path.basename(model_id)}.mp4", fps=16)


if __name__ == "__main__":
    main()
