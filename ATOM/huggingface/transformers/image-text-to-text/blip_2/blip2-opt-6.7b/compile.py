import os

from optimum.rbln import RBLNAutoModelForImageTextToText


def main():
    model_id = "Salesforce/blip2-opt-6.7b"
    model = RBLNAutoModelForImageTextToText.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "language_model": {
                "batch_size": 1,
                "max_seq_len": 2048,  # default "max_position_embeddings"
                "num_devices": 4,
                "use_inputs_embeds": True,
            },
        },
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
