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
                # `image_prefill_chunk_size` takes either a single value or a list of bucket sizes.
                # When a list is provided, multiple image prefill chunk sizes can be served using N models.
                # Please note the below conditions.
                # `max(image_prefill_chunk_size) >= max(max_soft_tokens)`
                # `min(image_prefill_chunk_size) >= min(max_soft_tokens)`
                # `all([ipcs % 128 == 0 for ipcs in image_prefill_chunk_size]) is True`
                # By default, `image_prefill_chunk_size = 384`.
                "image_prefill_chunk_size": [384, 640, 1152],
            },
            "vision_tower": {
                "num_devices": 8,
                # max_soft_tokens should be set to one of the values listed below, as the Gemma4
                # preprocessor supports only these five feature sizes.
                # By default, `max_soft_tokens = 280`.
                "max_soft_tokens": [70, 140, 280, 560, 1120],
            },
        },
    )
    model.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
