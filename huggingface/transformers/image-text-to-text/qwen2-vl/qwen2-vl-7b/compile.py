import os

from optimum.rbln import RBLNAutoModelForVision2Seq


def main():
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    model = RBLNAutoModelForVision2Seq.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            # The `device` parameter specifies the device allocation for each submodule during runtime.
            # As Qwen2-VL consists of multiple submodules, loading them all onto a single device may exceed its memory capacity, especially as the batch size increases.
            # By distributing submodules across devices, memory usage can be optimized for efficient runtime performance.
            "visual": {
                # Max sequence length for Vision Transformer (ViT), representing the number of patches in an image.
                # Example: For a 224x224 pixel image with patch size 14,
                # this produces 256 patches [(224/14) * (224/14)]. Thus, max_seq_lens must be at least 256.
                # RBLN optimization processes inference per image or video frame, so set max_seq_lens to
                # match the maximum expected resolution to optimize computation.
                "max_seq_lens": 6400,
            },
            "tensor_parallel_size": 8,
            "max_seq_len": 32_768,
        },
    )
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
