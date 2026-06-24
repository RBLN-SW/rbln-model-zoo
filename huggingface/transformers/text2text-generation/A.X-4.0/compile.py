import argparse
import os

from optimum.rbln import RBLNAutoModelForCausalLM


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        choices=["A.X-4.0-Light"],
        default="A.X-4.0-Light",
        help="(str) model type, Size of A.X-4.0. [A.X-4.0-Light]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"skt/{args.model_id}"

    # Compile and export
    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=16_384,  # default "max_position_embeddings"
        rbln_num_devices=4,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
