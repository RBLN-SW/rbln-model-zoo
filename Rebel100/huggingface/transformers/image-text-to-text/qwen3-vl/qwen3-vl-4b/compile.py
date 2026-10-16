import os

from optimum.rbln import RBLNAutoModelForImageTextToText


def main():
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    model = RBLNAutoModelForImageTextToText.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            # The `device` parameter specifies the device allocation for each submodule during runtime.
            # As Qwen3-VL consists of multiple submodules, loading them all onto a single device may exceed its memory capacity, especially as the batch size increases.
            # By distributing submodules across devices, memory usage can be optimized for efficient runtime performance.
            "visual": {
                # Maximum sequence lengths for Vision Transformer attention. Can be an integer or list of integers,
                # each indicating the number of patches in a sequence for an image or video. For example, an image
                # of 512x512 pixels with patch size 16 yields (512/16) * (512/16) = 1024 patches; the ViT
                # counts them before the 2x2 spatial merge, on the size after the processor's resize.
                # RBLN optimization runs inference per image, or per temporal patch (2 frames) of a video,
                # so set `max_seq_len` to match the maximum expected resolution to reduce computation.
                "max_seq_len": 16384,
                "num_devices": 1,
                "create_runtimes": False,
            },
            "num_devices": 1,
            "kvcache_partition_len": 8_192,
            # Max position embedding for the language model, must be a multiple of kvcache_partition_len.
            "max_seq_len": 262_144,
            "create_runtimes": False,
        },
    )
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
