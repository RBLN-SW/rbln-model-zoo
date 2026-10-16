import argparse
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "stable-fast-3d"))
)

from modeling_stable_fast_3d import RBLNSF3D


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-fast-3d",
        help="(str) Hugging Face model id or local path.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()

    model = RBLNSF3D.from_pretrained(
        model_id=args.model_id,
        export=True,
        rbln_batch_size=1,
        rbln_image_size=512,
    )

    output_dir = os.path.abspath("./sf3d")
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
