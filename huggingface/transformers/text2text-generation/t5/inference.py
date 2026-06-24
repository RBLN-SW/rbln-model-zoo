import argparse
import os

from optimum.rbln import RBLNAutoModelForSeq2SeqLM
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
        default="t5-base",
        help="(str) model type, Size of T5. [t5-small, t5-base, t5-large, t5-3b, t5-11b]",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="translate English to German: The house is wonderful.",
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = args.model_name
    text = args.text

    # Load compiled model
    model = RBLNAutoModelForSeq2SeqLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Generate tokens
    output = model.generate(
        **inputs,
        max_length=256,
    )

    # Show text and result
    print(f"Text: {text}")
    print(f"Result: {tokenizer.decode(output[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
