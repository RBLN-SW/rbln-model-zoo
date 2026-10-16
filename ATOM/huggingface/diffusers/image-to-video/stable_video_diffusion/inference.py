import argparse
import os

import torch
from diffusers.utils import export_to_video, load_image
from optimum.rbln import RBLNStableVideoDiffusionPipeline


def parsing_argument():
    parser = argparse.ArgumentParser(
        description="Compile Stable Video Diffusion model to RBLN format"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="stable-video-diffusion-img2vid",
        choices=["stable-video-diffusion-img2vid", "stable-video-diffusion-img2vid-xt"],
        help="Model ID of the Stable Video Diffusion model to compile",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"stabilityai/{args.model_name}"

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
    image = load_image(url)

    # Compile and export
    pipe = RBLNStableVideoDiffusionPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    height, width = pipe.vae.rbln_config.sample_size
    num_frames = pipe.unet.rbln_config.num_frames

    # Resize the image to make it suitable for the compiled model
    image = image.resize((width, height))

    # Generate video
    generator = torch.manual_seed(42)
    frames = pipe(
        image=image,
        height=height,
        width=width,
        num_frames=num_frames,
        generator=generator,
    ).frames[0]

    # Save image result
    export_to_video(
        frames, f"generated_{os.path.basename(model_id)}.mp4", fps=num_frames
    )


if __name__ == "__main__":
    main()
