import argparse
import os

from optimum.rbln import RBLNAutoPipelineForText2Image


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="red cat, 4k photo",
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "kandinsky-community/kandinsky-2-2-decoder"
    prompt = args.prompt

    # Load compiled model
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate image
    image = pipe(prompt, height=768, width=768, num_inference_steps=50).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
