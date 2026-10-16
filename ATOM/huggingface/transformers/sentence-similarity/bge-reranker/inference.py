import argparse
import os

from optimum.rbln import RBLNAutoModelForSequenceClassification
from transformers import AutoTokenizer

MAX_SEQ_LEN_CFG = {
    "v2-m3": 8192,
    "large": 512,
    "base": 512,
}


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["v2-m3", "large", "base"],
        default="v2-m3",
        help="(str) model type, Size of bge-reranker. [v2-m3, large, base]",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="what is panda?",
        help="(str) type, query context for score",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="The giant panda (Ailuropoda melanoleuca), "
        "sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        help="(str) type, messege context for score",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"BAAI/bge-reranker-{args.model_name}"

    # Load compiled model
    model = RBLNAutoModelForSequenceClassification.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(
        args.query,
        args.message,
        padding="max_length",
        return_tensors="pt",
        max_length=MAX_SEQ_LEN_CFG[args.model_name],
    )

    # run model
    output = model(inputs.input_ids, inputs.attention_mask)[0]

    # get score
    score = output.view(-1).float()

    # Show text and result
    print(f"Query: {args.query}")
    print(f"Message: {args.message}")
    print(f"Score: {score}")


if __name__ == "__main__":
    main()
