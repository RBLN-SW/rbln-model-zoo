import argparse
import os

import torch
from optimum.rbln import RBLNAutoPipelineForText2Image


def parsing_argument():
    parser = argparse.ArgumentParser()
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
    guidance_scale = args.guidance_scale

    # Compile and export
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_guidance_scale=guidance_scale,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
