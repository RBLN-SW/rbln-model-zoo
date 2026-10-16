import os

import torch
from optimum.rbln import RBLNAutoPipelineForText2Image


def main():
    # Base model ID from Huggingface Hub
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    # LoRA model IDs:
    # 1. LCM (Latent Consistency Model) LoRA for faster inference
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
    # 2. Pixel Art style LoRA
    pixel_lora_id = "nerijs/pixel-art-xl"

    # Initialize the RBLN pipeline for SDXL with multiple LoRA adaptations
    # NOTE: Unlike GPU-based pipelines where LoRAs can be loaded dynamically after model creation,
    # RBLN SDK requires all LoRA configurations at compile time since the adapters need to be
    # fused with the weights during compilation for NPU inference
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        # Enable model export/compilation for NPU
        export=True,
        # LoRA configurations must be specified during model loading for RBLN compilation
        lora_ids=[lcm_lora_id, pixel_lora_id],
        # Filenames of the LoRA weights within their repositories
        lora_weights_names=[
            "pytorch_lora_weights.safetensors",
            "pixel-art-xl.safetensors",
        ],
        # Scaling factors for each LoRA's effect (higher = stronger effect)
        lora_scales=[1.0, 1.2],
        # RBLN-specific configuration:
        # Set UNet batch size to 2 for handling both conditional and unconditional predictions
        # required for classifier-free guidance during inference
        rbln_config={"unet": {"batch_size": 2}},
    )

    # Below is the traditional GPU-based method of loading LoRAs (NOT supported in RBLN SDK)
    # Dynamic loading approach cannot be used with RBLN as LoRAs must be fused during compilation
    """
    pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")
    pipe.load_lora_weights("./pixel-art-xl.safetensors", adapter_name="pixel")
    pipe.set_adapters(["lora", "pixel"], adapter_weights=[1.0, 1.2])
    """

    # Save the compiled model with fused LoRA weights to a local directory
    pipe.save_pretrained(f"{os.path.basename(model_id)}-pixel-lora")


if __name__ == "__main__":
    main()
