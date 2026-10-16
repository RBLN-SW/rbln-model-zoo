import os

from optimum.rbln import RBLNAutoModelForSeq2SeqLM


def main():
    model_id = "gogamza/kobart-summarization"

    # Compile and export
    model = RBLNAutoModelForSeq2SeqLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_enc_max_seq_len=1024,
        rbln_dec_max_seq_len=1024,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
