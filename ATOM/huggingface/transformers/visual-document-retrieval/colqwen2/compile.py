import os

from optimum.rbln import RBLNColQwen2ForRetrieval


def main():
    model_id = "vidore/colqwen2-v1.0-hf"

    # Compile and export the model
    model = RBLNColQwen2ForRetrieval.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_config={
            # The `device` parameter specifies the device allocation for each submodule during runtime.
            # As ColQwen2_5ForRetrieval consists of multiple submodules, loading them all onto a single device may exceed its memory capacity, especially as the batch size increases.
            # By distributing submodules across devices, memory usage can be optimized for efficient runtime performance.
            "vlm": {
                "visual": {
                    # Max sequence length for Vision Transformer (ViT), representing the number of patches in an image.
                    "max_seq_len": 6400,
                },
                "num_devices": 4,
                # Max position embedding for the language model, must be a multiple of kvcache_partition_len.
                "max_seq_len": 32_768,
            },
        },
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
