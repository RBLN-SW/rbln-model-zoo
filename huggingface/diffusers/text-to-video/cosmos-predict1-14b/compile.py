import os

from cosmos_upsampler import (
    RBLNMistralNeMoForTextUpsampler,
    RBLNMistralNeMoForTextUpsamplerConfig,
)
from optimum.rbln import RBLNAutoConfig, RBLNAutoModel, RBLNCosmosTextToWorldPipeline


def main():
    # step 1. Compile Cosmos Upsampler
    upsampler_model_id = "nvidia/Cosmos-UpsamplePrompt1-12B-Text2World"

    # Register Custom Class
    RBLNAutoModel.register(RBLNMistralNeMoForTextUpsampler, exist_ok=True)
    RBLNAutoConfig.register(RBLNMistralNeMoForTextUpsamplerConfig, exist_ok=True)

    text_upsampler = RBLNMistralNeMoForTextUpsampler.from_pretrained(
        model_id=upsampler_model_id,
        export=True,
        rbln_config={
            "batch_size": 1,
            "max_seq_len": 1024,
            "num_devices": 4,
            "create_runtimes": False,
        },
    )
    # Save upsampler compiled results to disk
    text_upsampler.save_pretrained(os.path.basename(upsampler_model_id))

    # step 2. Compile Cosmos core Pipeline (transformer, text_encoder, vae)
    model_id = "nvidia/Cosmos-1.0-Diffusion-14B-Text2World"

    # By default, the generated video is a 4-second clip with a resolution of 704x1280 pixels and a frame rate of 30 frames per second (fps).
    # The video content visualizes the input text description as a short animated scene, capturing key elements within the specified time constraints.
    # Aspect ratios and resolutions are configurable, with options including 1:1 (960x960 pixels), 4:3 (960x704 pixels), 3:4 (704x960 pixels),
    # 16:9 (1280x704 pixels), and 9:16 (704x1280 pixels).
    height = 704
    width = 1280

    pipe = RBLNCosmosTextToWorldPipeline.from_pretrained(
        model_id,
        export=True,  # export PyTorch models to RBLN models with optimum
        rbln_config={
            "height": height,
            "width": width,
            "create_runtimes": False,
            "transformer": {
                "num_devices": 4,
            },
        },
    )

    # Save all compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
