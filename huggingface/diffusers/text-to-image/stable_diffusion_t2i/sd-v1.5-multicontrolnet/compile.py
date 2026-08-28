import argparse
import os

import torch
from diffusers import ControlNetModel, UniPCMultistepScheduler
from optimum.rbln import RBLNAutoPipelineForText2Image


def parsing_argument():
    parser = argparse.ArgumentParser()
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
    model_id = "benjamin-paine/stable-diffusion-v1-5"
    controlnet_model_ids = [
        "lllyasviel/sd-controlnet-openpose",
        "lllyasviel/sd-controlnet-canny",
    ]

    controlnets = []
    for cmi in controlnet_model_ids:
        controlnet = ControlNetModel.from_pretrained(cmi, torch_dtype=torch.float16)
        controlnets.append(controlnet)

    # Compile and export
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        export=True,  # export a PyTorch model to RBLN model with optimum
        controlnet=controlnets,
        rbln_config={
            "img_width": args.img_width,
            "img_height": args.img_height,
            "unet": {"batch_size": 2},
        },
        scheduler=UniPCMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        ),
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
