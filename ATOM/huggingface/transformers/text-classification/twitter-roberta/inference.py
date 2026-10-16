import argparse
import csv
import os
import urllib.request

import numpy as np
import torch
from optimum.rbln import RBLNAutoModelForSequenceClassification
from transformers import RobertaTokenizerFast


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        default="Celebrating my promotion 😎",
        help="(str) type, text for score",
    )
    return parser.parse_args()


# Preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def download_label_mapping(task):
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode("utf-8").split("\n")
        csvreader = csv.reader(html, delimiter="\t")
    return [row[1] for row in csvreader if len(row) > 1]


def main():
    args = parsing_argument()
    task = "emotion"
    model_id = f"cardiffnlp/twitter-roberta-base-{task}"

    # Load compiled model
    model = RBLNAutoModelForSequenceClassification.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = RobertaTokenizerFast.from_pretrained(os.path.basename(model_id))
    labels = download_label_mapping(task)

    # Encode the text
    text = preprocess(args.text)
    inputs = tokenizer(
        [text],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Run the model
    output = model(inputs.input_ids, inputs.attention_mask)[0]

    # Show text and result
    print(f"Text: {args.text}")

    num_class = 4
    for batch_itr in range(output.shape[0]):
        # Apply softmax to get probabilities
        scores = output[batch_itr].detach()
        scores = torch.nn.functional.softmax(scores, dim=-1).numpy()

        # Get ranking of scores
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        # Print out the results
        for i in range(scores.shape[0]):
            label = labels[ranking[i]]
            score = scores[ranking[i]]
            print(f"{batch_itr}) {label} {np.round(float(score), num_class)}")


if __name__ == "__main__":
    main()
