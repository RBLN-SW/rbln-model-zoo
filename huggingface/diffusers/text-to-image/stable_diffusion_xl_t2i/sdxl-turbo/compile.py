import os

import torch
from optimum.rbln import RBLNAutoPipelineForText2Image


def main():
    model_id = "stabilityai/sdxl-turbo"

    # Compile and export
    # As SDXL-turbo does not use guidance_scale, we disable them with rbln_guidance_scale=0.0
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_guidance_scale=0.0,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
