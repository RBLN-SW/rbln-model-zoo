import rebel  # RBLN Compiler
import torch
from sentence_transformers import SentenceTransformer


class RBLNLaBSEWrapper(torch.nn.Module):
    def __init__(self, model):
        super(RBLNLaBSEWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        output = self.model(inputs)
        return (
            output["token_embeddings"],
            output["sentence_embedding"],
        )


def main():
    model_id = "sentence-transformers/LaBSE"

    # Instantiate model from sentence-transformers
    model = SentenceTransformer(model_id)
    model = RBLNLaBSEWrapper(model)

    # Set `max sequence length` of the compiled model
    MAX_SEQ_LEN = 256

    # Compile
    input_info = [
        ("input_ids", [1, MAX_SEQ_LEN], "int64"),
        ("attention_mask", [1, MAX_SEQ_LEN], "int64"),
        ("token_type_ids", [1, MAX_SEQ_LEN], "int64"),
    ]

    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save("labse.rbln")


if __name__ == "__main__":
    main()
