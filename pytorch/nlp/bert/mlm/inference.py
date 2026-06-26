import argparse

import rebel  # RBLN Runtime
from transformers import BertTokenizer


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
    MAX_SEQ_LEN = 128

    # Prepare input text sequence for masked language modeling
    tokenizer = BertTokenizer.from_pretrained(model_name)
    text = "the color of rose is [MASK]."
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", max_length=MAX_SEQ_LEN
    )

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"bert-{args.model_name}.rbln", tensor_type="pt")

    # Run inference
    logits = module(**inputs)

    # Decoding final logit to text
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
        as_tuple=True
    )[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    print(f"Predicted token: {tokenizer.decode(predicted_token_id)}")


if __name__ == "__main__":
    main()
