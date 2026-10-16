import argparse
import os

from optimum.rbln import RBLNAutoModelForCausalLM
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        default='''def print_prime(n):
    """
    Print all primes between 1 and n
    """''',
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "microsoft/phi-2"

    # Load compiled model
    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(args.text, return_tensors="pt", padding=True)

    # Generate tokens
    output_sequence = model.generate(
        **inputs,
        max_length=200,
    )

    input_len = inputs.input_ids.shape[-1]
    generated_texts = tokenizer.decode(
        output_sequence[0][input_len:], skip_special_tokens=True, skip_prompt=True
    )

    # Show text and result
    print(f"Text: {args.text}")
    print(f"Result: {generated_texts}")


if __name__ == "__main__":
    main()
