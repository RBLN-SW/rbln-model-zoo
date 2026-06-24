import argparse
import os

from optimum.rbln import RBLNAutoModelForCausalLM
from transformers import GPT2Tokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        default="gpt2-xl",
        help="(str) model type, Size of GPT2. [gpt2, gpt2-medium, gpt2-large, gpt2-xl]",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Replace me with any text you'd like.",
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    text = args.text
    model_id = f"openai-community/{args.model_name}"

    # Load compiled model
    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.pad_token_id = 50256

    inputs = tokenizer(text, return_tensors="pt")

    # Generate tokens
    output = model.generate(
        **inputs,
        max_length=26,
    )

    # Show input and result
    print(f"Text: {text}")
    print(f"Result: {tokenizer.decode(output.numpy()[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
