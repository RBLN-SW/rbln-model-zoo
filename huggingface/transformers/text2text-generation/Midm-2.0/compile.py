import argparse
import os

from optimum.rbln import RBLNAutoModelForCausalLM

DEFAULT_NUM_DEVICES = {
    "Midm-2.0-Base-Instruct": 8,
    "Midm-2.0-Mini-Instruct": 4,
}


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        choices=["Midm-2.0-Base-Instruct", "Midm-2.0-Mini-Instruct"],
        default="Midm-2.0-Base-Instruct",
        help="(str) model type, Size of midm. [Midm-2.0-Base-Instruct, Midm-2.0-Mini-Instruct]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()

    model_id = f"K-intelligence/{args.model_id}"

    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,
        rbln_batch_size=1,
        rbln_max_seq_len=32_768,
        rbln_num_devices=DEFAULT_NUM_DEVICES[os.path.basename(model_id)],
    )

    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
