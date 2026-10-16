import os

import torch
from optimum.rbln import RBLNCosmos2TextToImagePipeline


def main():
    model_id = "nvidia/Cosmos-Predict2-14B-Text2Image"

    # By default, Cosmos-Predict2 Text2Image generates a single 768x1360 image.
    height = 768
    width = 1360

    pipe = RBLNCosmos2TextToImagePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        export=True,  # export PyTorch models to RBLN models with optimum
        rbln_config={
            "height": height,
            "width": width,
            "create_runtimes": False,
            "text_encoder": {
                "num_devices": 4,
            },
            "transformer": {
                "num_devices": 8,
            },
        },
    )

    # Save all compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
