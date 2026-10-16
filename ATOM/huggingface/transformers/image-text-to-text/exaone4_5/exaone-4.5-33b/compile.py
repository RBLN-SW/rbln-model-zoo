import os

from optimum.rbln import RBLNAutoModelForImageTextToText


def main():
    model_id = "LGAI-EXAONE/EXAONE-4.5-33B"
    model = RBLNAutoModelForImageTextToText.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            # For LGAI-EXAONE/EXAONE-4.5-33B, longest_edge set 28 * 28 * 4096 in preprocessor_config.json
            "visual": {
                # Max sequence length for Vision Transformer (ViT), representing the number of patches in an image.
                # Example: For a 224x196 pixel image with patch size 14 and window size 112,
                # the width is padded to 224, resulting in a 224x224 image.
                # This produces 256 patches [(224/14) * (224/14)]. Thus, max_seq_len must be at least 256.
                # For window-based attention, max_seq_len must be a multiple of (window_size / patch_size)^2, e.g., (112/14)^2 = 64.
                # Hence, 256 (64 * 4) is valid. RBLN optimization processes inference per image or video frame, so set max_seq_len to
                # match the maximum expected resolution to optimize computation.
                "max_seq_len": 16384,
                "num_devices": 16,
            },
            "num_devices": 16,
            "kvcache_partition_len": 8192,
            # Max position embedding for the language model, must be a multiple of kvcache_partition_len.
            "max_seq_len": 262144,
        },
    )
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
