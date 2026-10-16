import argparse
import os

import torch
from diffusers.utils import export_to_video, load_video
from optimum.rbln import RBLNCosmos2_5_TransferPipeline

DEFAULT_PROMPT = (
    "The video is a driving scene through a modern urban environment, likely captured from a dashcam or a "
    "similar fixed camera setup inside a vehicle. The scene unfolds on a wide, multi-lane road flanked by "
    "tall, modern buildings with glass facades. The road is relatively empty, with only a few cars visible, "
    "including a black car directly ahead of the camera, maintaining a steady pace. The camera remains "
    "static, providing a consistent view of the road and surroundings as the vehicle moves forward.On the "
    "left side of the road, there are several trees lining the sidewalk, providing a touch of greenery "
    "amidst the urban setting. Pedestrians are visible on the sidewalks, some walking leisurely, while "
    "others stand near the buildings. The buildings are a mix of architectural styles, with some featuring "
    "large glass windows and others having more traditional concrete exteriors. A few commercial signs and "
    "logos are visible on the buildings, indicating the presence of businesses and offices.Traffic cones "
    "are placed on the road ahead, suggesting some form of roadwork or lane closure, guiding the vehicles "
    "to merge or change lanes. The road markings are clear, with white arrows indicating the direction of "
    "travel. The sky is clear, suggesting a sunny day, which enhances the visibility of the scene. "
    "Throughout the video, the vehicle maintains a steady speed, and the camera captures the gradual "
    "approach towards the intersection, where the road splits into different directions. The overall "
    "atmosphere is calm and orderly, typical of a city during non-peak hours."
)


def parsing_argument():
    parser = argparse.ArgumentParser(
        description="Run Cosmos Transfer2.5 2B (edge control)"
    )
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
    model_id = "nvidia/Cosmos-Transfer2.5-2B"

    # Control video: the pre-computed Canny edge maps for the car driving scene from the
    # official cosmos-transfer2.5 assets (the prompt above is the matching car_prompt.txt).
    controls = [
        load_video(
            "https://github.com/nvidia-cosmos/cosmos-transfer2.5/raw/refs/heads/main/assets/car_example/edge/car_edge.mp4"
        )
    ]

    # Load all pipeline compiled model
    pipe = RBLNCosmos2_5_TransferPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        rbln_config={
            # The `rbln_config` is a dictionary used to pass configurations for the model and its submodules.
            # The `device` parameter specifies which device should be used for each submodule during runtime.
            #
            # Since Cosmos-Transfer2.5 consists of multiple submodules, loading them all onto a
            # single device may exceed its memory capacity. Therefore, when creating runtimes for
            # each submodule, devices can be divided and assigned for efficient memory utilization.
            #
            # For example — approximate per-submodule memory footprints (rebel-compiler 0.11.3),
            # used to plan the device assignment below. Assume ~15.7 GiB per device (e.g. RBLN-CA25):
            #   text_encoder     ~1.9 GiB per device (8 devices)
            #   transformer      ~5.7 GiB on device 0, ~1.9 GiB on the others (8 devices)
            #   controlnet       ~5.6 GiB + ~3.3 GiB (2 devices)
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
            "controlnet": {
                "device": [0, 1],
            },
            "vae": {
                "device_map": {"encoder": 2, "decoder": 3},
            },
            "safety_checker": {
                "qwen3guard": {"device": 0},
                "face_blur_filter": {"device": 0},
            },
        },
    )

    print(f"Input prompt: {args.text}")

    # Transfer inference with controls: the output video follows the edge maps. The total
    # length defaults to the control video (121 frames here), generated auto-regressively in
    # 93-frame chunks on the host; the chunk size is pinned to the compiled num_frames.
    video = pipe(
        controls=controls,
        controls_conditioning_scale=1.0,
        prompt=args.text,
        height=704,
        width=1280,
        generator=torch.Generator().manual_seed(args.seed),
    ).frames[0]
    export_to_video(video, f"{os.path.basename(model_id)}.mp4", fps=16)


if __name__ == "__main__":
    main()
