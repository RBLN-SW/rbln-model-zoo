import argparse
import os

import numpy as np
from diffusers.utils import load_image
from optimum.rbln import RBLNKandinskyV22InpaintPipeline, RBLNKandinskyV22PriorPipeline
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
    prior_model_id = "kandinsky-community/kandinsky-2-2-prior"
    decoder_model_id = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
    prompt = args.prompt

    # Load compiled model
    prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(
        model_id=os.path.basename(prior_model_id),
        export=False,
    )
    decoder_pipe = RBLNKandinskyV22InpaintPipeline.from_pretrained(
        model_id=os.path.basename(decoder_model_id),
        export=False,
    )

    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
    init_image = load_image(img_url)
    # Resize the image to make it suitable for the compiled model
    init_image = init_image.resize((768, 768), resample=Image.BICUBIC, reducing_gap=1)

    image_emb, zero_image_emb = prior_pipe(prompt, return_dict=False)
    width, height = init_image.size

    # Mask out the desired area to inpaint
    # In this example, we will draw a hat on the cat's head
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:250, 250:-250] = 1

    # Generate image
    image = decoder_pipe(
        image=init_image,
        mask_image=mask,
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        height=768,
        width=768,
    ).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
