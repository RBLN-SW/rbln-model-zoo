import os

from optimum.rbln import RBLNAutoModelForVision2Seq


def main():
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    model = RBLNAutoModelForVision2Seq.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            # The `device` parameter specifies the device allocation for each submodule during runtime.
            # As Qwen3-VL consists of multiple submodules, loading them all onto a single device may exceed its memory capacity, especially as the batch size increases.
            # By distributing submodules across devices, memory usage can be optimized for efficient runtime performance.
            "visual": {
                # Maximum sequence lengths for Vision Transformer attention. Can be an integer or list of integers,
                # each indicating the number of patches in a sequence for an image or video. For example, an image
                # of 224x224 pixels with patch size 16 and spatial_merge_size 2 yields
                # (224/16/2) * (224/16/2) = 49 merged patches. RBLN optimization runs inference
                # per image or video frame, so set `max_seq_len` to match the maximum expected
                # resolution to reduce computation.
                "max_seq_lens": 16384,
                "create_runtimes": False,
            },
            "tensor_parallel_size": 8,
            "kvcache_partition_len": 16_384,
            # Max position embedding for the language model, must be a multiple of kvcache_partition_len.
            "max_seq_len": 262_144,
            "create_runtimes": False,
        },
    )
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
