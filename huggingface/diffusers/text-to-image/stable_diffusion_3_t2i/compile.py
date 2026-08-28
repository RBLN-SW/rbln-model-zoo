import os

import torch
from optimum.rbln import RBLNAutoPipelineForText2Image


def main():
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    # Compile and export
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_img_height=1024,
        rbln_img_width=1024,
        rbln_guidance_scale=7.0,
        rbln_config={
            # The `rbln_config` is a dictionary used to pass configurations for the model and its submodules.
            # The `device` parameter specifies which device should be used for each submodule during runtime.
            #
            # Since Stable Diffusion consists of multiple submodules, loading all submodules onto a single device may occasionally exceed its memory capacity.
            # Therefore, when creating runtimes for each submodule, devices can be divided and assigned to ensure efficient memory utilization.
            #
            # For example:
            # - Assume each device has a memory capacity of 15.7 GiB (e.g., RBLN-CA12).
            # - `text_encoder` (~192 MB), `text_encoder_2` (~1.2 GiB), `transformer` (~4.1 GiB), and `VAE` (~2.0 GiB) are assigned to device 0,
            #     which totals approximately ~7.4 GiB, comfortably fitting within the available memory of Device 0.
            # - `text_encoder_3` (~8.7 GiB) is assigned to device 1 to prevent an Out Of Memory (OOM) error on device 0.
            "text_encoder": {"device": 0},
            "text_encoder_2": {"device": 0},
            "text_encoder_3": {"device": 1},
            "transformer": {"device": 0},
            "vae": {"device": 0},
        },
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
