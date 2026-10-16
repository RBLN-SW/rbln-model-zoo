import argparse
import os

from diffusers.utils import load_image
from optimum.rbln import RBLNAutoPipelineForImage2Image


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",  # noqa: E501
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    prompt = args.prompt

    # Load compiled model
    pipe = RBLNAutoPipelineForImage2Image.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        rbln_config={
            # The `rbln_config` is a dictionary used to pass configurations for the model and its submodules.
            # The `device` parameter specifies which device should be used for each submodule during runtime.
            # Since Stable Diffusion consists of multiple submodules, loading all submodules onto a single device may occasionally exceed its memory capacity.
            # Therefore, when creating runtimes for each submodule, devices can be divided and assigned to ensure efficient memory utilization.
            # For example:
            # - Assume each device has a memory capacity of 15.7 GiB (e.g., RBLN-CA12).
            # - `text_encoder` (~192 MB), `text_encoder_2` (~1.2 GiB), `transformer` (~4.1 GiB), and `VAE` (~2.0 GiB) are assigned to device 0,
            #     which totals approximately ~7.4 GiB, comfortably fitting within the available memory of Device 0.
            # - `text_encoder_3` (~8.7 GiB) is assigned to device 1 to prevent an Out Of Memory (OOM) error on device 0.
            "text_encoder": {"device": 0},
            "text_encoder_2": {"device": 0},
            "text_encoder_3": {"device": 1},
            "transformer": {"device": 0},
            "vae": {"device": 0},
        },
    )

    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    init_image = load_image(url).resize((1024, 1024))

    # Generate image
    image = pipe(prompt, image=init_image, strength=0.95, guidance_scale=7.5).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
