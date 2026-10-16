import argparse
import os

from diffusers.utils import load_image
from optimum.rbln import RBLNAutoPipelineForImage2Image


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, Pixar, Disney",
        help="(str) type, prompt for generate image",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=512,
        help="input image width for generation",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=512,
        help="input image height for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "stabilityai/sdxl-turbo"
    prompt = args.prompt

    # Load compiled model
    pipe = RBLNAutoPipelineForImage2Image.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    init_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    ).resize((args.img_height, args.img_width))

    # Generate image
    image = pipe(
        prompt,
        image=init_image,
        num_inference_steps=2,
        guidance_scale=0.0,
        strength=0.5,
    ).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
