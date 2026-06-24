import argparse
import os

import torch
import torch.nn.functional as F
from optimum.rbln import RBLNAutoModelForTextEncoding
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default="query: how much protein should a female eat",
        help="(str) type, input query context",
    )
    parser.add_argument(
        "--passage",
        type=str,
        default="passage: As a general guideline, the CDC's average requirement of protein "
        "for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, "
        "you'll need to increase that if you're expecting or training for a marathon. "
        "Check out the chart below to see how much protein you should be eating each day.",
        help="(str) type, input messege context",
    )
    return parser.parse_args()


def average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_position_ids(
    input_ids: torch.Tensor,
    max_original_positions: int = 512,
    encode_max_length: int = 4096,
) -> torch.Tensor:
    position_ids = list(range(input_ids.size(1)))
    factor = max(encode_max_length // max_original_positions, 1)
    if input_ids.size(1) <= max_original_positions:
        position_ids = [(pid * factor) for pid in position_ids]

    position_ids = torch.tensor(position_ids, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    return position_ids


def main():
    args = parsing_argument()
    model_id = "dwzhu/e5-base-4k"

    # Load compiled model
    model = RBLNAutoModelForTextEncoding.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_q = tokenizer(
        [args.query],
        max_length=4096,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_p = tokenizer(
        [args.passage],
        max_length=4096,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_q["position_ids"] = get_position_ids(
        input_q["input_ids"], max_original_positions=512, encode_max_length=4096
    ).contiguous()
    input_p["position_ids"] = get_position_ids(
        input_p["input_ids"], max_original_positions=512, encode_max_length=4096
    ).contiguous()

    # run model
    q_output = model(**input_q)[0]
    p_output = model(**input_p)[0]

    embeddings_q = average_pool(q_output[0], input_q["attention_mask"])
    embeddings_p = average_pool(p_output[0], input_p["attention_mask"])

    embeddings_q = F.normalize(embeddings_q, p=2, dim=1)
    embeddings_p = F.normalize(embeddings_p, p=2, dim=1)

    # get similarity score
    score = (embeddings_q @ embeddings_p.T) * 100

    # Show text and result
    print(f"Query: {args.query}")
    print(f"Passage: {args.passage}")
    print(f"Score: {score.tolist()}")


if __name__ == "__main__":
    main()
