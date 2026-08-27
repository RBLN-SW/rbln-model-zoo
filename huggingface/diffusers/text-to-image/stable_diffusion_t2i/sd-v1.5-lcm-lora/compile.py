import os

import torch
from optimum.rbln import RBLNAutoPipelineForText2Image


def main():
    # Base model ID from Huggingface Hub
    # lykon/dreamshaper-7 is a model that has been fine-tuned on runwayml/stable-diffusion-v1-5.
    model_id = "Lykon/dreamshaper-7"

    # LoRA model IDs: LCM (Latent Consistency Model) LoRA for faster inference
    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

    # Compile and export
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        # Enable model export/compilation for NPU
        export=True,
        # LoRA configurations must be specified during model loading for RBLN compilation
        lora_ids=lcm_lora_id,
        lora_weights_names="pytorch_lora_weights.safetensors",
        rbln_guidance_scale=0.0,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id) + "-lora")


if __name__ == "__main__":
    main()
