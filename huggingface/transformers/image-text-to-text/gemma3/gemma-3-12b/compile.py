import os

from optimum.rbln import RBLNAutoModelForImageTextToText


def main():
    model_id = "google/gemma-3-12b-it"
    model_dir = os.path.basename(model_id)

    model = RBLNAutoModelForImageTextToText.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "language_model": {
                "num_devices": 8,
                "kvcache_partition_len": 16_384,
                "use_inputs_embeds": True,
            }
        },
    )
    model.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
