import argparse
import os

from diffusers.utils import load_image
from optimum.rbln import RBLNAutoPipelineForInpainting


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",  # noqa: E501
        help="(str) type, prompt for generate image",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="guidance_scale for sdxl-turbo",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    prompt = args.prompt
    guidance_scale = args.guidance_scale

    # Load compiled model
    pipe = RBLNAutoPipelineForInpainting.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
    mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
    source = load_image(img_url)
    mask = load_image(mask_url)

    # Generate image
    image = pipe(
        prompt, image=source, mask_image=mask, guidance_scale=guidance_scale
    ).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
