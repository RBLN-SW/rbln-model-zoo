import argparse

import rebel  # RBLN Compiler
from transformers import BertForMaskedLM


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
    model_name = "bert-" + args.model_name + "-uncased"
    # Set `max sequence length` of the compiled model
    MAX_SEQ_LEN = 128

    # Instantiate HuggingFace PyTorch BERT-base model
    model = BertForMaskedLM.from_pretrained(model_name, return_dict=False).eval()

    # Compile
    input_info = [
        ("input_ids", [1, MAX_SEQ_LEN], "int64"),
        ("attention_mask", [1, MAX_SEQ_LEN], "int64"),
        ("token_type_ids", [1, MAX_SEQ_LEN], "int64"),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"bert-{args.model_name}.rbln")


if __name__ == "__main__":
    main()
