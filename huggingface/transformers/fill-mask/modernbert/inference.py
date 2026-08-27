import argparse
import os

from optimum.rbln import RBLNAutoModelForMaskedLM
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["base", "large"],
        default="base",
        help="(str) type, Size of ModernBERT. [base or large]",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Plants create [MASK] through a process known as photosynthesis.",
        help="(str) type, text for score",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="(int) type, number of top predictions",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "answerdotai/ModernBERT-" + args.model_name

    # Load compiled model and tokenizer
    model = RBLNAutoModelForMaskedLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # ModernBERT takes only input_ids and attention_mask (no token_type_ids)
    inputs = tokenizer(
        args.text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    output = model(inputs.input_ids, inputs.attention_mask)[0]

    # Predict the masked words in the sentence
    masked_index = (inputs.input_ids.squeeze() == tokenizer.mask_token_id).nonzero()

    print("--- text ---")
    print(args.text)
    print("--- predictions ---")
    for mask_index in masked_index:
        mask_logits = output.squeeze()[mask_index.item()]
        probs = mask_logits.softmax(dim=-1)
        values, predictions = probs.topk(args.top_k)
        for value, prediction in zip(values.tolist(), predictions.tolist()):
            print(f"{tokenizer.decode([prediction]).strip()}: {value:.5f}")


if __name__ == "__main__":
    main()
