import argparse
import os

import numpy as np
import torch
from diffusers.utils import load_image
from optimum.rbln import RBLNAutoModelForDepthEstimation, RBLNAutoPipelineForImage2Image
from PIL import Image
from transformers import DPTImageProcessor


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="A robot, 4k photo",
        help="(str) type, prompt for generate image",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=0.5,
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    prompt = args.prompt
    controlnet_conditioning_scale = args.controlnet_conditioning_scale

    feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

    image = load_image(
        "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        "/kandinsky/cat.png"
    ).resize((1024, 1024))

    def get_depth_map(image, depth_estimator):
        image = feature_extractor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            depth_map = depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    # Load compiled model
    pipe = RBLNAutoPipelineForImage2Image.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )
    dpt = RBLNAutoModelForDepthEstimation.from_pretrained(
        model_id="dpt-hybrid-midas", export=False
    )
    depth_image = get_depth_map(image, dpt)

    # Generate image
    new_image = pipe(
        prompt,
        image=image,
        control_image=depth_image,
        strength=0.99,
        num_inference_steps=50,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images[0]

    # Save image result
    new_image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
