import argparse
import os

import torch
from optimum.rbln import RBLNAutoPipelineForImage2Image


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10.5,
        help="guidance_scale for sdxl-turbo",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=1024,
        help="input image width for generation",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=1024,
        help="input image height for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    guidance_scale = args.guidance_scale

    # Compile and export
    pipe = RBLNAutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_guidance_scale=guidance_scale,
        rbln_img_width=args.img_width,
        rbln_img_height=args.img_height,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
