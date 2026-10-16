import argparse
import os

from optimum.rbln import RBLNAutoModelForQuestionAnswering


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["base", "large"],
        default="base",
        help="(str) type, Size of BERT. [base or large]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = (
        "deepset/bert-base-cased-squad2"
        if args.model_name == "base"
        else "deepset/bert-large-uncased-whole-word-masking-squad2"
    )

    # Compile and export
    model = RBLNAutoModelForQuestionAnswering.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_max_seq_len=512,  # default "max_position_embedding"
        rbln_batch_size=1,
    )
    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
