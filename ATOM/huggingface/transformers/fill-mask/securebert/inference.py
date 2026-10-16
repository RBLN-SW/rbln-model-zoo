import argparse
import os

import torch
from optimum.rbln import RBLNAutoModelForMaskedLM
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        default="Gathering this information may reveal opportunities for other forms of <mask>, "
        "establishing operational resources, or initial access.",
        help="(str) type, text for score",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "ehsanaghaei/SecureBERT"

    model = RBLNAutoModelForMaskedLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Function to predict the masked words in a sentence
    inputs = tokenizer(
        args.text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    masked_position = (inputs.input_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position]
    words = []

    output = model(inputs.input_ids, inputs.attention_mask)[0]

    last_hidden_state = output[0].squeeze()

    # Show text and result
    print(f"Text: {args.text}")

    list_of_list = []
    for _, mask_index in enumerate(masked_pos):
        mask_hidden_state = last_hidden_state[mask_index]
        idx = torch.topk(mask_hidden_state, k=10, dim=0)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        words = [w.replace(" ", "") for w in words]
        list_of_list.append(words)
        print(f"Predictions: {words}")

    best_guess = ""
    for j in list_of_list:
        best_guess = best_guess + "," + j[0]

    return words


if __name__ == "__main__":
    main()
