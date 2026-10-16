import argparse
import os

from diffusers.utils import export_to_video, load_video
from optimum.rbln import (
    RBLNAutoModelForImageTextToText,
    RBLNCosmosVideoToWorldPipeline,
)
from transformers import AutoProcessor


def parsing_argument():
    parser = argparse.ArgumentParser(description="Run Cosmos Video2World 7B")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="(str) type, Text prompt for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    video = load_video(
        "https://github.com/nvidia-cosmos/cosmos-predict1/raw/refs/heads/main/assets/diffusion/video2world_input1.mp4"
    )

    if args.text is None:
        print(
            "Text prompt for generation is not provided. The prompt will be generated with the Video2World Prompt Upsampler (Pixtral-12B)."
        )
        upsampler_model_id = "mistral-community/pixtral-12b"
        upsampler = RBLNAutoModelForImageTextToText.from_pretrained(
            model_id=os.path.basename(upsampler_model_id),
            export=False,
            rbln_config={
                "vision_tower": {
                    "device": 1,
                },
                "language_model": {
                    "device": [0, 1, 2, 3],
                },
            },
        )
        processor = AutoProcessor.from_pretrained(upsampler_model_id)
        template = (
            "[INST][IMG]\n"
            + "Your task is to transform a given prompt into a refined and concise video description, no more than 150 words. Focus only on the content, no filler words or descriptions on the style. Never mention things outside the video.[/INST]"
        )
        inputs = processor(images=video[-1], text=template, return_tensors="pt")

        generated_ids = upsampler.generate(**inputs, max_new_tokens=400)
        prompt = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    else:
        prompt = args.text

    print(f"Input prompt: {prompt}")

    model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Video2World"

    # Load all pipeline compiled model
    pipe = RBLNCosmosVideoToWorldPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        rbln_config={
            # The `rbln_config` is a dictionary used to pass configurations for the model and its submodules.
            # The `device` parameter specifies which device should be used for each submodule during runtime.
            #
            # Since Cosmos VideoToWorld consists of multiple submodules, loading all submodules onto a single device may occasionally exceed its memory capacity.
            # Therefore, when creating runtimes for each submodule, devices can be divided and assigned to ensure efficient memory utilization.
            #
            # For example — approximate per-submodule memory footprints (rebel-compiler 0.11.0),
            # used to plan the device assignment below. Assume ~15.7 GiB per device (e.g. RBLN-CA22):
            #   upsampler        ~6.7 GB   (4 devices)
            #   text_encoder     ~9.2 GB
            #   transformer      ~7.9 GB (1 device) / ~5.1 GB (3 devices)
            #   VAE encoder      ~5.1 GB
            #   VAE decoder      ~6.1 GB
            #   safety_checker   qwen3guard ~1.3 GB, face_blur_filter ~150 MB
            #
            # As of cosmos-guardrail 0.3.1 the video guardrail keeps only `face_blur_filter`;
            # the SigLIP content-safety filter is disabled upstream, so `siglip_encoder`
            # and `video_safety_model` are no longer compiled.
            "transformer": {
                "device": [0, 1, 2, 3],
            },
            "text_encoder": {
                "device": 4,
            },
            "vae": {
                "device_map": {"encoder": 4, "decoder": 5},
            },
            "safety_checker": {
                "qwen3guard": {"device": 5},
                "face_blur_filter": {"device": 5},
            },
        },
    )

    # Inference with video & text pair and Generate video from them
    output = pipe(video=video, prompt=prompt).frames[0]
    export_to_video(output, f"{os.path.basename(model_id)}.mp4", fps=30)


if __name__ == "__main__":
    main()
