import argparse
import os

from optimum.rbln import RBLNAutoModelForSeq2SeqLM
from transformers import PegasusTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/pegasus-xsum",
        help="(str) model name and target task",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="PG&E stated it scheduled the blackouts in response to forecasts for high winds "
        "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
        "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.",
        help="(str) type, text for summarization",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    tokenizer = PegasusTokenizer.from_pretrained(args.model_name)

    # Load compiled model
    model = RBLNAutoModelForSeq2SeqLM.from_pretrained(
        model_id=os.path.basename(args.model_name),
        export=False,
    )

    # Prepare inputs
    inputs = tokenizer(args.text, return_tensors="pt", padding="max_length")

    generation_kwargs = {}
    generation_kwargs["num_beams"] = 1
    generation_kwargs["max_length"] = model.generation_config.max_length

    # Generate tokens
    output_sequence = model.generate(
        **inputs,
        **generation_kwargs,
    )

    # Show text and result
    print(f"Text: {args.text}")
    print(f"Result: {tokenizer.decode(output_sequence[0])}")


if __name__ == "__main__":
    main()
