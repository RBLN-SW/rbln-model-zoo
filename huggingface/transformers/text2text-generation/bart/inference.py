import argparse
import os

from optimum.rbln import RBLNAutoModelForSeq2SeqLM
from transformers import BartTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["base", "large"],
        default="base",
        help="(str) model type, Size of BART [base, large]",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="UN Chief Says There Is No <mask> in Syria",
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"facebook/bart-{args.model_name}"

    tokenizer = BartTokenizer.from_pretrained(model_id)

    # Load compiled model

    # [info] `facebook/bart-base` and `facebook/bart-large` are
    # primarily designed for feature extraction.
    # For text-to-text generation (fill-mask task) in vLLM,
    # it is recommended to use the BartForConditionalGeneration model
    # (ex. `lucadiliello/bart-small` model).
    model = RBLNAutoModelForSeq2SeqLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    inputs = tokenizer(args.text, return_tensors="pt", padding="max_length")

    generation_kwargs = {}
    generation_kwargs["num_beams"] = 1

    # Generate tokens
    output_sequence = model.generate(
        **inputs,
        max_length=1024,
        **generation_kwargs,
    )

    # Show text and result
    print(f"Text: {args.text}")
    print(f"Result: {tokenizer.decode(output_sequence[0])}")


if __name__ == "__main__":
    main()
