import argparse
import os

from optimum.rbln import RBLNAutoModelForQuestionAnswering
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["base", "large"],
        default="base",
        help="(str) type, Size of BERT. [base or large]",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is Rebellions?",
        help="(str) type, question text",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="Rebellions is the best NPU company.",
        help="(str) type, context text",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = (
        "deepset/bert-base-cased-squad2"
        if args.model_name == "base"
        else "deepset/bert-large-uncased-whole-word-masking-squad2"
    )

    # Load compiled model
    model = RBLNAutoModelForQuestionAnswering.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Tokenize to the compiled fixed sequence length (rbln_max_seq_len=512).
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(
        args.question,
        args.context,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    # Decode the answer span from the start/end logits.
    outputs = model(**inputs)
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    answer_tokens = inputs["input_ids"][0, answer_start_index : answer_end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # Result
    print(f"Question: {args.question}")
    print(f"Context: {args.context}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
