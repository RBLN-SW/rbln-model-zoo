import argparse
import os

from optimum.rbln import RBLNAutoModelForTextEncoding


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["large", "base", "small"],
        default="large",
        help="(str) model type, Size of bge-en-v1.5. [large, base, small]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"BAAI/bge-{args.model_name}-en-v1.5"

    # Compile and export
    model = RBLNAutoModelForTextEncoding.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=512,  # default "max_position_embeddings"
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
