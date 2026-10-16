import argparse
import os

from optimum.rbln import RBLNAutoPipelineForText2Image


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        help="(str) type, prompt for generate image",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="guidance_scale for sdxl-base",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    prompt = args.prompt
    guidance_scale = args.guidance_scale

    # Load compiled model
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate image
    image = pipe(prompt, guidance_scale=guidance_scale).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
