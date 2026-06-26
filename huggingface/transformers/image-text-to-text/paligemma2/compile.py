import os

from optimum.rbln import RBLNAutoModelForImageTextToText


def main():
    model_id = "google/paligemma2-3b-mix-224"
    model = RBLNAutoModelForImageTextToText.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "language_model": {
                "batch_size": 1,
                "max_seq_len": 8192,  # default "max_position_embeddings"
                "num_devices": 4,
                "prefill_chunk_size": 8192,
            },
        },
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
