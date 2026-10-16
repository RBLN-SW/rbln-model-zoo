import os

from optimum.rbln import RBLNAutoModelForImageTextToText


def main():
    model_id = "google/gemma-4-31B-it"
    model_dir = os.path.basename(model_id)

    model = RBLNAutoModelForImageTextToText.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "batch_size": 1,
            "language_model": {
                "num_devices": 16,
                "max_seq_len": 262_144,
                "kvcache_partition_len": 16384,
                "prefill_chunk_size": 128,
                "attn_impl": "flash_attn",
            },
            "vision_tower": {
                "num_devices": 8,
            },
        },
    )
    model.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
