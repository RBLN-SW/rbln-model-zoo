import os

import torch
from diffusers import AutoModel
from optimum.rbln import RBLNCosmos2_5_TransferPipeline


def main():
    model_id = "nvidia/Cosmos-Transfer2.5-2B"

    # Cosmos-Transfer2.5 is the Predict2.5 base pipeline plus a per-modality ControlNet.
    # The official Hub layout keeps them on separate revisions, so the ControlNet variant
    # (edge, depth, seg, or blur) is loaded on its own and passed to the pipeline.
    controlnet = AutoModel.from_pretrained(
        model_id,
        revision="diffusers/controlnet/general/edge",
        torch_dtype=torch.bfloat16,
    )

    # Long videos are generated auto-regressively in `num_frames`-sized chunks, so the
    # compiled shapes are per chunk (93 frames, 704x1280) while the total output length
    # follows the control video.
    height = 704
    width = 1280
    num_frames = 93

    pipe = RBLNCosmos2_5_TransferPipeline.from_pretrained(
        model_id,
        revision="diffusers/general",
        controlnet=controlnet,
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
            "controlnet": {
                "num_devices": 2,
            },
        },
    )

    # Save all compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
