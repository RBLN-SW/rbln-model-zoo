import argparse
import os

from diffusers.utils import load_image
from optimum.rbln import RBLNAutoPipelineForImage2Image


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="A red cartoon frog, 4k",
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "kandinsky-community/kandinsky-2-2-decoder"
    prompt = args.prompt

    # Load compiled model
    pipe = RBLNAutoPipelineForImage2Image.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/frog.png"
    init_image = load_image(img_url)

    # Generate image
    image = pipe(
        args.prompt,
        image=init_image,
        height=768,
        width=768,
        num_inference_steps=100,
        strength=0.2,
    ).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
