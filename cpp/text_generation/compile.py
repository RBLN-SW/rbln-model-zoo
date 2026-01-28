import os
from argparse import ArgumentParser

from optimum.rbln import RBLNLlamaForCausalLM


def parsing_argument() -> object:
    parser = ArgumentParser()
    parser.add_argument(
        "--model-id",
        dest="model_id",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model id",
    )
    return parser.parse_args()


def main() -> None:
    args = parsing_argument()

    model = RBLNLlamaForCausalLM.from_pretrained(
        model_id=args.model_id,
        export=True,
        rbln_batch_size=1,
        rbln_max_seq_len=8192,
        rbln_tensor_parallel_size=4,
    )

    # Standalone example behavior: write to a local directory.
    model.save_pretrained(os.path.basename(args.model_id))


if __name__ == "__main__":
    main()
