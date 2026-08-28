import argparse
import os

import torch
from optimum.rbln import RBLNAutoPipelineForImage2Image


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_height",
        type=int,
        default=512,
        help="input image width for generation",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=768,
        help="input image height for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "benjamin-paine/stable-diffusion-v1-5"

    # Compile and export
    pipe = RBLNAutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_img_width=args.img_width,
        rbln_img_height=args.img_height,
        rbln_config={
            "img_width": args.img_width,
            "img_height": args.img_height,
            "unet": {"batch_size": 2},
        },
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
