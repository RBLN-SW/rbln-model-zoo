import os

from optimum.rbln import RBLNT5EncoderModel


def main():
    model_id = "sentence-transformers/sentence-t5-xxl"

    # Compile and export
    model = RBLNT5EncoderModel.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=512,  # default "max_position_embeddings"
        rbln_num_devices=4,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
