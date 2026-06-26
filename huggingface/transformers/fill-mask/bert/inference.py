import argparse
import os

import numpy as np
import torch
from optimum.rbln import RBLNAutoModelForMaskedLM
from transformers import BertTokenizer


# Function to score and predict the masked words in a sentence
# ref: https://github.com/huggingface/transformers/blob/6b550462139655d488d4c663086a63e98713c6b9/src/transformers/pipelines/fill_mask.py#L131
def postprocess(tokenizer, model_outputs, batch_input_ids, top_k=5):
    masked_indices = [
        torch.nonzero(input_ids == tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
        for input_ids in batch_input_ids
    ]

    logits = [
        output[masked_index, :]
        for output, masked_index in zip(model_outputs, masked_indices)
    ]
    probs = [logit.softmax(dim=-1) for logit in logits]

    values = [prob.topk(top_k)[0] for prob in probs]
    predictions = [prob.topk(top_k)[1] for prob in probs]
    result_batch = []
    single_masks = [value.shape[0] == 1 for value in values]
    for input_ids, value, prediction, masked_index, single_mask in zip(
        batch_input_ids, values, predictions, masked_indices, single_masks
    ):
        result = []
        for i, (_value, _prediction) in enumerate(
            zip(value.tolist(), prediction.tolist())
        ):
            row = []
            for v, p in zip(_value, _prediction):
                # Copy is important since we're going to modify this array in place
                tokens = input_ids.numpy().copy()

                tokens[masked_index[i]] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != tokenizer.pad_token_id)]
                # Originally we skip special tokens to give readable output.
                # For multi masks though, the other [MASK] would be removed otherwise
                # making the output look odd, so we add them back
                sequence = tokenizer.decode(tokens, skip_special_tokens=single_mask)
                proposition = {
                    "score": v,
                    "token": p,
                    "token_str": tokenizer.decode([p]),
                    "sequence": sequence,
                }
                row.append(proposition)
            result.append(row)
        if single_mask:
            result_batch.append(result[0])
        else:
            result_batch.append(result)
    return result_batch


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["base", "large"],
        default="base",
        help="(str) type, Size of BERT. [base or large]",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello I'm a [MASK] model.",
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
    model_id = "google-bert/bert-" + args.model_name + "-uncased"

    # Load compiled model and tokenizer
    model = RBLNAutoModelForMaskedLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    tokenizer = BertTokenizer.from_pretrained(model_id)

    # Function to predict the masked words in a sentence
    inputs = tokenizer(
        args.text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = tokenizer(
        args.text, return_tensors="pt", padding="max_length", max_length=512
    )
    output = model(inputs.input_ids, inputs.attention_mask, inputs.token_type_ids)[0]
    results = postprocess(tokenizer, output, inputs.input_ids, top_k=args.top_k)

    prediction = [
        [result[i]["token_str"] for i in range(args.top_k)] for result in results
    ]
    print(f"Tokens: {prediction}")
    return results


if __name__ == "__main__":
    main()
