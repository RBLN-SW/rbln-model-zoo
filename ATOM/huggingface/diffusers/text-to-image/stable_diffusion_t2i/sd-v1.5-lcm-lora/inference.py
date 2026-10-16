import os

from diffusers import LCMScheduler
from optimum.rbln import RBLNAutoPipelineForText2Image


def main():
    # Base model ID (used here just for constructing the compiled model path)
    model_id = "Lykon/dreamshaper-7"

    # Load the previously compiled model with fused LoRA weights
    model_path = f"{os.path.basename(model_id)}-lora"

    # Load compiled model
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id=model_path,
        export=False,
    )

    # Define the prompt for image generation
    prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

    # Replace the default scheduler with LCM scheduler
    # LCM scheduler enables faster inference with fewer steps while maintaining quality
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Generate image using the compiled model
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
    ).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
