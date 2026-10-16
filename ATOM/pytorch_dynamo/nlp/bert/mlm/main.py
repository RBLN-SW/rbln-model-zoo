import argparse
import os

import rebel  # noqa: F401  # needed to use torch dynamo's "rbln" backend
import torch
from transformers import BertForMaskedLM, BertTokenizer


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

    # Instantiate HuggingFace PyTorch BERT model
    model = BertForMaskedLM.from_pretrained(model_name)

    # Compile the model using torch.compile with RBLN backend
    model = torch.compile(
        model,
        backend="rbln",
        # Disable dynamic shape support, as the RBLN backend currently does not support it
        dynamic=False,
        options={"cache_dir": f"./rbln_{os.path.basename(model_name)}"},
    )

    # Prepare input text sequence for masked language modeling
    tokenizer = BertTokenizer.from_pretrained(model_name)
    text = "the color of rose is [MASK]."
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", max_length=MAX_SEQ_LEN
    )

    # (Optional) First call of forward invokes the compilation
    model(**inputs)

    # Run inference using the compiled model
    logits = model(**inputs).logits

    # Decoding final logit to text
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
        as_tuple=True
    )[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    print(f"Predicted word: {tokenizer.decode(predicted_token_id)}")


if __name__ == "__main__":
    main()
