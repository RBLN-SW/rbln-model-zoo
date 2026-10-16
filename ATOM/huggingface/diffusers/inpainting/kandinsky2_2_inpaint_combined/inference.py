import argparse
import os

import numpy as np
from diffusers.utils import load_image
from optimum.rbln import RBLNAutoPipelineForInpainting
from PIL import Image


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="a hat",
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
    prompt = args.prompt

    # Load compiled model
    pipe = RBLNAutoPipelineForInpainting.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
    init_image = load_image(img_url)
    # Resize the image to make it suitable for the compiled model
    init_image = init_image.resize((768, 768), resample=Image.BICUBIC, reducing_gap=1)
    width, height = init_image.size

    # Mask out the desired area to inpaint
    # In this example, we will draw a hat on the cat's head
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:250, 250:-250] = 1

    # Generate image
    image = pipe(prompt, image=init_image, mask_image=mask).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
