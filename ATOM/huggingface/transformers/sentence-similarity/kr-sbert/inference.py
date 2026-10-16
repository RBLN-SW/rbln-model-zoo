import argparse
import os

import torch
import torch.nn.functional as F
from optimum.rbln import RBLNAutoModelForTextEncoding
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sentence_0",
        type=str,
        default="This is an example sentence",
        help="(str) type, first input sentence",
    )
    parser.add_argument(
        "--sentence_1",
        type=str,
        default="Each sentence is converted",
        help="(str) type, second input sentence",
    )
    return parser.parse_args()


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def main():
    args = parsing_argument()
    model_id = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

    # Load compiled model
    model = RBLNAutoModelForTextEncoding.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_0 = tokenizer(
        [args.sentence_0],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=True,
    )
    input_1 = tokenizer(
        [args.sentence_1],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=True,
    )

    # run model
    output_0 = model(**input_0)
    output_1 = model(**input_1)

    # get similarity score
    vector_0 = mean_pooling(output_0, input_0["attention_mask"])
    vector_1 = mean_pooling(output_1, input_1["attention_mask"])

    # this part is same as sentence_transformers util.cos_sim
    # ref: https://github.com/UKPLab/sentence-transformers/blob/969082a485960b9f4f377ed62be942637af24121/sentence_transformers/util.py#L92
    vector_0 = F.normalize(vector_0, p=2, dim=1)
    vector_1 = F.normalize(vector_1, p=2, dim=1)
    score = torch.mm(vector_0, vector_1.transpose(0, 1))

    # Show text and result
    print(f"Sentence 0: {args.sentence_0}")
    print(f"Sentence 1: {args.sentence_1}")
    print(f"Score: {score.tolist()}")


if __name__ == "__main__":
    main()
