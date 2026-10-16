import os

import torch
from optimum.rbln import RBLNCosmos2VideoToWorldPipeline


def main():
    model_id = "nvidia/Cosmos-Predict2-14B-Video2World"

    # Cosmos-Predict2 Video2World generates a 93-frame clip with a resolution of 704x1280
    # pixels at 16 fps, conditioned on an input image or video that enters through the VAE
    # encoder at runtime.
    height = 704
    width = 1280
    num_frames = 93

    pipe = RBLNCosmos2VideoToWorldPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        export=True,  # export PyTorch models to RBLN models with optimum
        rbln_config={
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "create_runtimes": False,
            "transformer": {
                "num_devices": 8,
            },
            "text_encoder": {
                "num_devices": 4,
            },
        },
    )

    # Save all compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
