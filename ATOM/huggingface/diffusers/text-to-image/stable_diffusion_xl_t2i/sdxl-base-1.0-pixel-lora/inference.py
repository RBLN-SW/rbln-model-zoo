import os

from diffusers import LCMScheduler
from optimum.rbln import RBLNAutoPipelineForText2Image


def main():
    # Base model ID (used here just for constructing the compiled model path)
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    # Load the previously compiled model with fused LoRA weights
    model_path = f"{os.path.basename(model_id)}-pixel-lora"
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_path,
        # Set export=False since we're loading an already compiled model
        export=False,
    )

    # Replace the default scheduler with LCM scheduler
    # LCM scheduler enables faster inference with fewer steps while maintaining quality
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Define the prompt for image generation
    prompt = "pixel, a cute corgi"
    negative_prompt = "3d render, realistic"

    # Generate image using the compiled model
    # LCM allows for much fewer steps than traditional schedulers (typically 4-8 steps)
    # Lower guidance scale (1.0-2.0) works better with LCM
    img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=8,
        guidance_scale=1.5,
    ).images[0]

    # Save the generated image
    img.save("lcm_lora.png")


if __name__ == "__main__":
    main()
