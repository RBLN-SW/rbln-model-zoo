import argparse
import os

from optimum.rbln import RBLNAutoModelForCausalLM


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        default="gpt2-xl",
        help="(str) model type, Size of GPT2. [gpt2, gpt2-medium, gpt2-large, gpt2-xl]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"openai-community/{args.model_name}"

    # Compile and export
    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=1024,  # default "n_positions"
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
