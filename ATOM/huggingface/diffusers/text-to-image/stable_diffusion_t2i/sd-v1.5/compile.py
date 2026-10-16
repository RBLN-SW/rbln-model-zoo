import os

import torch
from optimum.rbln import RBLNAutoPipelineForText2Image


def main():
    model_id = "benjamin-paine/stable-diffusion-v1-5"

    # Compile and export
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_config={"unet": {"batch_size": 2}},
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
