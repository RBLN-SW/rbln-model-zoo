import os

import torch
from optimum.rbln import RBLNCosmos2_5_PredictBasePipeline


def main():
    model_id = "nvidia/Cosmos-Predict2.5-2B"

    # Cosmos-Predict2.5 unifies Text2World, Image2World, and Video2World in a single pipeline.
    # The generated video is a 93-frame clip with a resolution of 704x1280 pixels at 16 fps.
    # Conditioning media (an image or a video, when given) enters through the VAE encoder at
    # runtime, so one compiled pipeline serves all three modes.
    height = 704
    width = 1280
    num_frames = 93

    pipe = RBLNCosmos2_5_PredictBasePipeline.from_pretrained(
        model_id,
        revision="diffusers/base/post-trained",
        torch_dtype=torch.bfloat16,
        export=True,  # export PyTorch models to RBLN models with optimum
        rbln_config={
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "max_seq_len": 512,
            "create_runtimes": False,
            "text_encoder": {
                "num_devices": 8,
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
