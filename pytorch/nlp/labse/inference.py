import argparse

import rebel
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        default="The weather is lovely today.",
        help="(str) type, text for getting embeddings",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "sentence-transformers/LaBSE"

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime("labse.rbln", tensor_type="pt")

    # Set `max sequence length` of the compiled model
    MAX_SEQ_LEN = 256

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=MAX_SEQ_LEN)
    inputs = tokenizer(
        args.text, padding="max_length", return_tensors="pt", max_length=MAX_SEQ_LEN
    )

    # run model
    embeddings = module.run(
        inputs.input_ids, inputs.attention_mask, inputs.token_type_ids
    )[1]

    # Show result
    print(f"Sentence embeddings: {embeddings}")


if __name__ == "__main__":
    main()
