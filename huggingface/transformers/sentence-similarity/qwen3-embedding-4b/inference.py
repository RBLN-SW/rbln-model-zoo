import os

import torch
import torch.nn.functional as F
from optimum.rbln import RBLNAutoModel
from torch import Tensor
from transformers import AutoTokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def main():
    model_id = "Qwen/Qwen3-Embedding-4B"

    # Load compiled model
    model = RBLNAutoModel.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    input_texts = queries + documents
    max_length = 8192

    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # run model
    hidden_states = []
    for i in range(len(batch_dict["input_ids"])):
        inputs = {
            "input_ids": batch_dict["input_ids"][i].unsqueeze(0),
            "attention_mask": batch_dict["attention_mask"][i].unsqueeze(0),
        }
        hidden_states.append(model(**inputs).last_hidden_state)

    hidden_states = torch.cat(hidden_states, dim=0)
    embeddings = last_token_pool(hidden_states, batch_dict["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = embeddings[:2] @ embeddings[2:].T

    # Show text and result
    print("--- query ---")
    print(queries)
    print("--- passage ---")
    print(documents)
    print("--- score ---")
    print(scores.tolist())


if __name__ == "__main__":
    main()
