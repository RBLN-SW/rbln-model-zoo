import os

from optimum.rbln import RBLNAutoModelForImageTextToText


def main():
    model_id = "mistral-community/pixtral-12b"
    model = RBLNAutoModelForImageTextToText.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "vision_tower": {
                "batch_size": 1,
                "output_hidden_states": True,
            },
            "language_model": {
                "num_devices": 8,
                "use_inputs_embeds": True,
                "batch_size": 1,
                "max_seq_len": 131_072,
                "kvcache_partition_len": 16_384,
            },
        },
    )
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
