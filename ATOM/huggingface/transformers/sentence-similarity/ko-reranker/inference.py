import argparse
import os

import numpy as np
from optimum.rbln import RBLNAutoModelForSequenceClassification
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        default="나는 너를 싫어해",
        help="(str) type, first text for score",
    )
    parser.add_argument(
        "--text_pair",
        type=str,
        default="나는 너를 사랑해",
        help="(str) type, second text for score",
    )
    return parser.parse_args()


def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


def main():
    args = parsing_argument()
    model_id = "Dongjin-kr/ko-reranker"

    # Load compiled model
    model = RBLNAutoModelForSequenceClassification.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pairs = [[args.text, args.text_pair]]

    # Prepare inputs
    inputs = tokenizer(
        pairs,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Run the model
    outputs = model(inputs.input_ids, inputs.attention_mask)[0]

    # Get score
    scores = exp_normalize(outputs.view(-1).numpy())

    # Show texts and result
    print(f"Texts: {pairs}")
    print(f"Score: {scores}")


if __name__ == "__main__":
    main()
