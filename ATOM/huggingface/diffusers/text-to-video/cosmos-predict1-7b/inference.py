import argparse
import os

from cosmos_upsampler import (
    RBLNMistralNeMoForTextUpsampler,
    RBLNMistralNeMoForTextUpsamplerConfig,
)
from diffusers.utils import export_to_video
from optimum.rbln import RBLNAutoConfig, RBLNAutoModel, RBLNCosmosTextToWorldPipeline
from transformers import AutoTokenizer

UPSAMPLE_TOKEN_THRESHOLD = 200


def parsing_argument():
    parser = argparse.ArgumentParser(description="Run Cosmos Text2World 7B")
    parser.add_argument(
        "--text",
        type=str,
        default="From the pedestrian's viewpoint, the scene unfolds on a snow-dusted street at dusk, where the sky is painted in deep purples and blues. Silver SUVs glide swiftly along a gentle bend, their headlights cutting through the softly falling snowflakes. The road is lined with parked vehicles, their silhouettes creating a narrow passage for the moving traffic. The snowflakes dance in the air, illuminated by the streetlights, adding a serene yet brisk atmosphere to the scene.",
        help="(str) type, Text prompt for generation",
    )
    return parser.parse_args()


def text_upsampling(text_upsampler, prompt, tokenizer, max_gen_len=512):
    dialogs = [
        [
            {
                "role": "user",
                "content": f"Upsample the short caption to a long caption: {prompt}",
            }
        ]
    ]

    prompt_tokens = [
        tokenizer.apply_chat_template(
            dialog, add_generation_prompt=True, tokenize=False
        )
        for dialog in dialogs
    ]
    inputs = tokenizer(prompt_tokens, return_tensors="pt", padding=True)

    input_len = inputs.input_ids.shape[-1]
    output_sequence = text_upsampler.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=min(1024, input_len + max_gen_len),
    )
    extended_prompt = tokenizer.decode(
        output_sequence[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).replace('"', "")

    return extended_prompt


def main():
    args = parsing_argument()
    model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
    upsampler_model_id = "nvidia/Cosmos-UpsamplePrompt1-12B-Text2World"

    # Register Custom Class
    RBLNAutoModel.register(RBLNMistralNeMoForTextUpsampler, exist_ok=True)
    RBLNAutoConfig.register(RBLNMistralNeMoForTextUpsamplerConfig, exist_ok=True)

    # Prepare upsampler tokenizer
    tokenizer = AutoTokenizer.from_pretrained(upsampler_model_id, use_fast=True)
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 10

    # Load upsampler compiled model
    text_upsampler = RBLNMistralNeMoForTextUpsampler.from_pretrained(
        model_id=os.path.basename(upsampler_model_id),
        export=False,
        rbln_config={
            "device": [0, 1, 2, 3],
        },
    )

    # Load all pipeline compiled model
    pipe = RBLNCosmosTextToWorldPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        rbln_config={
            # The `rbln_config` is a dictionary used to pass configurations for the model and its submodules.
            # The `device` parameter specifies which device should be used for each submodule during runtime.
            #
            # Since Cosmos TextToWorld consists of multiple submodules, loading all submodules onto a single device may occasionally exceed its memory capacity.
            # Therefore, when creating runtimes for each submodule, devices can be divided and assigned to ensure efficient memory utilization.
            #
            # For example — approximate per-submodule memory footprints (rebel-compiler 0.11.0),
            # used to plan the device assignment below. Assume ~15.7 GiB per device (e.g. RBLN-CA22):
            #   text_upsampler   ~5.6 GB   (4 devices)
            #   text_encoder     ~9.2 GB
            #   transformer      ~7.9 GB (1 device) / ~5.1 GB (3 devices)
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
                "device": 4,
            },
            "safety_checker": {
                "qwen3guard": {"device": 3},
                "face_blur_filter": {"device": 3},
            },
        },
    )

    # Running Guardrail for input text
    print(f"Input prompt: {args.text}")
    if not pipe.safety_checker.check_text_safety(args.text):
        raise ValueError(
            f"Cosmos Guardrail detected unsafe text in the prompt: {args.text}. Please ensure that the "
            f"prompt abides by the NVIDIA Open Model License Agreement."
        )

    # Check token length is lower than thershold and Running Upsampler
    token_length = len(tokenizer.encode(args.text))
    if token_length <= UPSAMPLE_TOKEN_THRESHOLD:
        print(
            f"The input text contains fewer tokens ({token_length}) than "
            + f"the required threshold ({UPSAMPLE_TOKEN_THRESHOLD}). Upsampling will now begin."
        )
        upsampled_prompt = text_upsampling(text_upsampler, args.text, tokenizer)
        print(f"Upsampled prompt: {upsampled_prompt}")
    else:
        upsampled_prompt = args.text

    # Inference with upsampled text and Generate video from text
    output = pipe(prompt=upsampled_prompt).frames[0]
    export_to_video(output, f"{os.path.basename(model_id)}.mp4", fps=30)


if __name__ == "__main__":
    main()
