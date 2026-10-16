import os

import torch
from optimum.rbln import RBLNAutoPipelineForInpainting


def main():
    model_id = "kandinsky-community/kandinsky-2-2-decoder-inpaint"

    # Compile and export
    pipe = RBLNAutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_config={
            "img_height": 768,
            "img_width": 768,
            "unet": {"batch_size": 2},
        },
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
