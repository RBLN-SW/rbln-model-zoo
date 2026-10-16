import argparse
import os

from diffusers.utils import load_image
from optimum.rbln import RBLNAutoPipelineForImage2Image


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="a dog catching a frisbee in the jungle",
        help="(str) type, prompt for generate image",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10.5,
        help="guidance_scale for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    prompt = args.prompt
    guidance_scale = args.guidance_scale

    # Load compiled model
    pipe = RBLNAutoPipelineForImage2Image.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    init_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
    )

    # Generate image
    image = pipe(
        prompt, image=init_image, guidance_scale=guidance_scale, strength=0.8
    ).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
