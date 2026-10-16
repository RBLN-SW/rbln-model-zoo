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
    model_name = (
        "deepset/bert-base-cased-squad2"
        if args.model_name == "base"
        else "deepset/bert-large-uncased-whole-word-masking-squad2"
    )
    MAX_SEQ_LEN = 384

    # Prepare input text sequence for masked language modeling
    tokenizer = BertTokenizer.from_pretrained(model_name)
    question, text = "What is Rebellions?", "Rebellions is the best NPU company."
    inputs = tokenizer(
        question,
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_SEQ_LEN,
    )

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"bert-{args.model_name}.rbln", tensor_type="pt")

    # Run inference
    out = module.run(**inputs)

    # Decoding final logit to text
    answer_start_index = out[0].argmax()
    answer_end_index = out[1].argmax()
    predict_answer_tokens = inputs.input_ids[
        0, answer_start_index : answer_end_index + 1
    ]
    print(f"Answer: {tokenizer.decode(predict_answer_tokens)}")


if __name__ == "__main__":
    main()
