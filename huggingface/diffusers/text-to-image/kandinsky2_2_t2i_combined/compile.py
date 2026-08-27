import os

import torch
from optimum.rbln import RBLNAutoPipelineForText2Image


def main():
    model_id = "kandinsky-community/kandinsky-2-2-decoder"

    # Compile and export
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        rbln_config={
            "img_height": 768,
            "img_width": 768,
            "unet": {"batch_size": 2},
        },
        export=True,  # export a PyTorch model to RBLN model with optimum
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
