import os

import torch
from huggingface_hub import hf_hub_download
from optimum.rbln import RBLNT5EncoderModel
from safetensors.torch import load_file
from sentence_transformers import util
from transformers import AutoTokenizer


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(model_output)
    sum_embeddings = torch.sum(model_output * attention_mask_expanded, dim=1)
    sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
    sum_mask = torch.clamp(sum_mask, min=1e-9)

    return sum_embeddings / sum_mask


# This function is designed for sentence-transformers models, where a dense submodule is required in addition to the T5EncoderModel.
# The transformers-based optimum-rbln framework does not provide a dedicated class for this dense layer.
# Therefore, we define and compile the layer on-the-fly using torch.compile with the 'rbln' backend.
def load_and_compile_dense(repo_id: str, file_path: str):
    dense_ = torch.nn.Linear(in_features=1024, out_features=768, bias=False)
    local_file = hf_hub_download(repo_id=repo_id, filename=file_path)
    dense_.load_state_dict({"weight": load_file(local_file)["linear.weight"]})

    rbln_dense = torch.compile(
        dense_,
        backend="rbln",
        dynamic=False,
        options={"cache_dir": "./dense"},
    )

    return rbln_dense


def main():
    model_id = "sentence-transformers/sentence-t5-xxl"
    file_path = "2_Dense/model.safetensors"
    sentences = ["This is an example sentence", "Each sentence is converted"]

    # Load compiled model
    model = RBLNT5EncoderModel.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )
    dense_module = load_and_compile_dense(model_id, file_path)

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_0 = tokenizer(
        sentences[0],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    input_1 = tokenizer(
        sentences[1],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
    )

    # run T5Encoder model
    token_embeddings_0 = model(**input_0)[0]
    token_embeddings_1 = model(**input_1)[0]

    # mean pooling
    sentence_embeddings_0 = mean_pooling(token_embeddings_0, input_0["attention_mask"])
    sentence_embeddings_1 = mean_pooling(token_embeddings_1, input_1["attention_mask"])

    # Dense Module
    sentence_embeddings_0 = dense_module(sentence_embeddings_0[0])
    sentence_embeddings_1 = dense_module(sentence_embeddings_1[0])

    cosine_score = util.cos_sim(sentence_embeddings_0, sentence_embeddings_1).item()

    # Show text and result
    print(f"Sentence 0: {sentences[0]}")
    print(f"Sentence 1: {sentences[1]}")
    print(f"Score: {cosine_score}")


if __name__ == "__main__":
    main()
