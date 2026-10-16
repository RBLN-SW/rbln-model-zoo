import os

from optimum.rbln import RBLNAutoModelForMultimodalLM


def main():
    model_id = "Qwen/Qwen3-ASR-1.7B-hf"

    model = RBLNAutoModelForMultimodalLM.from_pretrained(
        model_id=model_id,
        export=True,
        rbln_config={
            "batch_size": 1,
            "max_seq_len": 65_536,
            "num_devices": 4,
            "attn_impl": "flash_attn",
            "kvcache_partition_len": 16_384,
            "audio_tower": {
                # Number of 8-second (104-token) attention windows the encoder graph handles per
                # NPU call, which is this checkpoint's default. Longer audio runs in more calls.
                "num_windows": 4,
            },
        },
    )

    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
