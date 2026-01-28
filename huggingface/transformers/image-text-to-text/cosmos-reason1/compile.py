import os

from optimum.rbln import RBLNAutoModelForVision2Seq


def main():
    model_id = "nvidia/Cosmos-Reason1-7B"
    model = RBLNAutoModelForVision2Seq.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            # The `device` parameter specifies the device allocation for each submodule during runtime.
            # As Qwen2.5-VL consists of multiple submodules, loading them all onto a single device may exceed its memory capacity, especially as the batch size increases.
            # By distributing submodules across devices, memory usage can be optimized for efficient runtime performance.
            "visual": {
                # Max sequence length for Vision Transformer (ViT), representing the number of patches in an image.
                # Example: For a 224x196 pixel image with patch size 14 and window size 112,
                # the width is padded to 224, resulting in a 224x224 image.
                # This produces 256 patches [(224/14) * (224/14)]. Thus, max_seq_len must be at least 256.
                # For window-based attention, max_seq_len must be a multiple of (window_size / patch_size)^2, e.g., (112/14)^2 = 64.
                # Hence, 256 (64 * 4) is valid. RBLN optimization processes inference per image or video frame, so set max_seq_len to
                # match the maximum expected resolution to optimize computation.
                "max_seq_lens": 8192,
                # The `device` parameter specifies which device should be used for each submodule during runtime.
            },
            "tensor_parallel_size": 8,
            "kvcache_partition_len": 16_384,
            # Max position embedding for the language model, must be a multiple of kvcache_partition_len.
            "max_seq_len": 114_688,
        },
    )
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
