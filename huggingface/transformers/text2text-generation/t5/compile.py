import argparse
import os

from optimum.rbln import RBLNAutoModelForSeq2SeqLM

DEFAULT_NUM_DEVICES = {
    "t5-small": 1,
    "t5-base": 1,
    "t5-large": 1,
    "t5-3b": 1,
    "t5-11b": 4,
}


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
        default="t5-base",
        help="(str) model type, Size of T5. [t5-small, t5-base, t5-large, t5-3b, t5-11b]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "google-t5/" + args.model_name

    # Compile and export
    model = RBLNAutoModelForSeq2SeqLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_num_devices=DEFAULT_NUM_DEVICES[args.model_name],
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
