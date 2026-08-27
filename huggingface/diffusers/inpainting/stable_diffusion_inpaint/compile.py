import os

import torch
from optimum.rbln import RBLNAutoPipelineForInpainting


def main():
    model_id = "stable-diffusion-v1-5/stable-diffusion-inpainting"

    # Compile and export
    pipe = RBLNAutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_guidance_scale=7.5,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
