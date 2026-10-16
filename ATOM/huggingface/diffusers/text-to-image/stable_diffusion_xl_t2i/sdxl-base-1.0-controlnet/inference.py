import argparse
import os

import cv2
import numpy as np
from diffusers.utils import load_image
from optimum.rbln import RBLNAutoPipelineForText2Image
from PIL import Image


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="aerial view, a futuristic research "
        "complex in a bright foggy jungle, hard lighting",
        help="(str) type, prompt for generate image",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, bad quality, sketches",
        help="(str) type, negative prompt for generate image",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=0.5,
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    controlnet_conditioning_scale = args.controlnet_conditioning_scale

    image = load_image(
        "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
    ).resize((1024, 1024))

    # input image preprocessing
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    # Load compiled model
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate image
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        image=canny_image,
    ).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
