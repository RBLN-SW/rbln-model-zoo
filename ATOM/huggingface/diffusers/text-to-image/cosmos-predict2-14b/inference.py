import argparse
import os

import torch
from optimum.rbln import RBLNCosmos2TextToImagePipeline

DEFAULT_PROMPT = (
    "A well-worn broom sweeps across a dusty wooden floor, its bristles gathering crumbs and flecks of debris "
    "in swift, rhythmic strokes. Dust motes dance in the sunbeams filtering through the window, glowing "
    "momentarily before settling. The quiet swish of straw brushing wood is interrupted only by the occasional "
    "creak of old floorboards. With each pass, the floor grows cleaner, restoring a sense of quiet order to "
    "the humble room."
)


def parsing_argument():
    parser = argparse.ArgumentParser(description="Run Cosmos Predict2 Text2Image 14B")
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
    model_id = "nvidia/Cosmos-Predict2-14B-Text2Image"

    # Load all pipeline compiled model
    pipe = RBLNCosmos2TextToImagePipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        rbln_config={
            # The `rbln_config` is a dictionary used to pass configurations for the model and its submodules.
            # The `device` parameter specifies which device should be used for each submodule during runtime.
            #
            # Since Cosmos-Predict2 Text2Image consists of multiple submodules, loading them all onto a
            # single device may exceed its memory capacity. Therefore, when creating runtimes for
            # each submodule, devices can be divided and assigned for efficient memory utilization.
            #
            # For example — approximate per-submodule memory footprints (rebel-compiler 0.11.3),
            # used to plan the device assignment below. Assume ~15.7 GiB per device (e.g. RBLN-CA25):
            #   text_encoder     ~2.4 GiB per device (4 devices, T5)
            #   transformer      ~4.0 GiB on device 0, ~3.8 GiB on device 4, ~3.7 GiB on the rest (8 devices)
            #   VAE decoder      ~12.5 GiB (no encoder is compiled for Text2Image)
            #   safety_checker   qwen3guard ~1.3 GiB, face_blur_filter ~150 MiB
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
                "device_map": {"decoder": 8},
            },
            "safety_checker": {
                "qwen3guard": {"device": 0},
                "face_blur_filter": {"device": 0},
            },
        },
    )

    print(f"Input prompt: {args.text}")

    # Generate image from text
    output = pipe(
        prompt=args.text,
        generator=torch.Generator().manual_seed(args.seed),
    ).images[0]
    output.save(f"{os.path.basename(model_id)}.png")


if __name__ == "__main__":
    main()
