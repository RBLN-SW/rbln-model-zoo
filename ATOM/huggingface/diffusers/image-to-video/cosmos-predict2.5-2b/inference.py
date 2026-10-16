import argparse
import os

import torch
from diffusers.utils import export_to_video, load_image
from optimum.rbln import RBLNCosmos2_5_PredictBasePipeline

DEFAULT_PROMPT = (
    "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple "
    "double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing "
    "forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights "
    "brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space "
    "in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, "
    "resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the "
    "vibrant, urban night scene."
)


def parsing_argument():
    parser = argparse.ArgumentParser(description="Run Cosmos Predict2.5 2B")
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
    model_id = "nvidia/Cosmos-Predict2.5-2B"

    # Conditioning image: the first frame the generated world video continues from.
    image = load_image(
        "https://media.githubusercontent.com/media/nvidia-cosmos/cosmos-predict2.5/refs/heads/main/assets/base/bus_terminal.jpg"
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
            #   transformer      ~3.0 GiB on device 0, ~1.8 GiB on the others (8 devices)
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
                "device_map": {"encoder": 0, "decoder": 1},
            },
            "safety_checker": {
                "qwen3guard": {"device": 2},
                "face_blur_filter": {"device": 2},
            },
        },
    )

    print(f"Input prompt: {args.text}")

    # Inference with image & text pair and Generate video from them
    video = pipe(
        image=image,
        prompt=args.text,
        num_frames=93,
        generator=torch.Generator().manual_seed(args.seed),
    ).frames[0]
    export_to_video(video, f"{os.path.basename(model_id)}.mp4", fps=16)


if __name__ == "__main__":
    main()
