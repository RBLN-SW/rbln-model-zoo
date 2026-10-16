import argparse
import os

import rebel  # noqa: F401  # needed to use torch dynamo's "rbln" backend
import torch
from transformers import BertForQuestionAnswering, BertTokenizer


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

    # Instantiate HuggingFace PyTorch BERT model
    model = BertForQuestionAnswering.from_pretrained(model_name)

    # Compile the model using torch.compile with RBLN backend
    model = torch.compile(
        model,
        backend="rbln",
        # Disable dynamic shape support, as the RBLN backend currently does not support it
        dynamic=False,
        options={"cache_dir": f"./{os.path.basename(model_name)}"},
    )

    # Prepare input text sequence for question answering
    tokenizer = BertTokenizer.from_pretrained(model_name)
    question, text = "What is Rebellions?", "Rebellions is the best NPU company."
    inputs = tokenizer(
        question,
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_SEQ_LEN,
    )

    # (Optional) First call of forward invokes the compilation
    model(**inputs)

    # Run inference using the compiled model
    outputs = model(**inputs)

    # Decoding final logits to text
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    predict_answer_tokens = inputs.input_ids[
        0, answer_start_index : answer_end_index + 1
    ]
    print(f"Question: {question}")
    print(f"Answer: {tokenizer.decode(predict_answer_tokens)}")


if __name__ == "__main__":
    main()
