import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TripoSR"))

from modeling_triposr import RBLNTripoSR


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/TripoSR",
        help="(str) Hugging Face model id or local path.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()

    model = RBLNTripoSR.from_pretrained(
        model_id=args.model_id,
        export=True,
        rbln_batch_size=1,
        rbln_image_size=512,
        rbln_chunk_size=8192,
    )

    output_dir = os.path.abspath("./triposr")
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
