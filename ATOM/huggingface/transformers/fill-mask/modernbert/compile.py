import argparse
import os

from optimum.rbln import RBLNAutoModelForMaskedLM


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["base", "large"],
        default="base",
        help="(str) type, Size of ModernBERT. [base or large]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "answerdotai/ModernBERT-" + args.model_name

    # Compile and export
    model = RBLNAutoModelForMaskedLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=512,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
