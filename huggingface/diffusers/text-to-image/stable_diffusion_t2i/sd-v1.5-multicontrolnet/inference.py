import argparse
import os

import cv2
import numpy as np
from controlnet_aux import OpenposeDetector
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from optimum.rbln import RBLNAutoPipelineForText2Image
from PIL import Image


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="a giant standing in a fantasy landscape, best quality",
        help="(str) type, prompt for generate image",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="monochrome, lowres, bad anatomy, worst quality, low quality",
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "benjamin-paine/stable-diffusion-v1-5"
    prompt = args.prompt
    negative_prompt = args.negative_prompt

    # input image preprocessing
    canny_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
    )
    canny_image = np.array(canny_image)

    low_threshold = 100
    high_threshold = 200

    canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)
    zero_start = canny_image.shape[1] // 4
    zero_end = zero_start + canny_image.shape[1] // 2
    canny_image[:, zero_start:zero_end] = 0

    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    openpose_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
    )
    openpose_image = openpose(openpose_image)

    # Load compiled model
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        scheduler=UniPCMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        ),
    )

    images = [openpose_image, canny_image]

    # Generate image
    image = pipe(
        prompt,
        images,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        controlnet_conditioning_scale=[1.0, 0.8],
    ).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
