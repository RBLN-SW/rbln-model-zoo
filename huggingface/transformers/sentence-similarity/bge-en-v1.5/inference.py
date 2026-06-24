import argparse
import os

import torch
from optimum.rbln import RBLNAutoModelForTextEncoding
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["large", "base", "small"],
        default="large",
        help="(str) model type, Size of bge-en-v1.5. [large, base, small]",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Embed this is sentence via Infinity.",
        help="(str) type, input query context",
    )
    parser.add_argument(
        "--passage",
        type=str,
        default="Paris is in France.",
        help="(str) type, input messege context",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"BAAI/bge-{args.model_name}-en-v1.5"

    # Load compiled model
    model = RBLNAutoModelForTextEncoding.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_q = tokenizer(
        args.query,
        padding="max_length",
        return_tensors="pt",
        max_length=512,
        return_token_type_ids=True,
    )
    input_p = tokenizer(
        args.passage,
        padding="max_length",
        return_tensors="pt",
        max_length=512,
        return_token_type_ids=True,
    )

    # run model
    q_output = model(input_q.input_ids, input_q.attention_mask, input_q.token_type_ids)
    p_output = model(input_p.input_ids, input_p.attention_mask, input_p.token_type_ids)
    q_embedding = torch.nn.functional.normalize(q_output[0][:, 0], p=2, dim=1)
    p_embedding = torch.nn.functional.normalize(p_output[0][:, 0], p=2, dim=1)

    # get similarity score
    score = q_embedding @ p_embedding.T

    # Show text and result
    print(f"Query: {args.query}")
    print(f"Passage: {args.passage}")
    print(f"Score: {score}")


if __name__ == "__main__":
    main()
