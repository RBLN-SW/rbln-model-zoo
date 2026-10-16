import argparse
import os

from optimum.rbln import RBLNAutoModelForSeq2SeqLM


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/pegasus-xsum",
        help="(str) model name and target task",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()

    # Compile and export
    model = RBLNAutoModelForSeq2SeqLM.from_pretrained(
        model_id=args.model_name,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(args.model_name))


if __name__ == "__main__":
    main()
