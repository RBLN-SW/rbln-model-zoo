import os

from optimum.rbln import RBLNAutoModelForCTC


def main():
    model_id = "facebook/wav2vec2-base-960h"

    # Compile and export
    model = RBLNAutoModelForCTC.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=160005,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
