import os

import torch
from optimum.rbln import RBLNColQwen2ForRetrieval
from PIL import Image
from transformers import ColQwen2Processor


def main():
    model_id = "vidore/colqwen2-v1.0-hf"

    # Load compiled model
    model = RBLNColQwen2ForRetrieval.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )
    processor = ColQwen2Processor.from_pretrained(model_id)

    # The document page screenshots from your corpus. Below are dummy images.
    images = [
        Image.new("RGB", (128, 128), color="white"),
        Image.new("RGB", (64, 32), color="black"),
    ]

    # The queries you want to retrieve documents for
    queries = [
        "When was the United States Declaration of Independence proclaimed?",
        "Who printed the edition of Romeo and Juliet?",
    ]

    # Process the inputs
    inputs_images = processor(images=images)
    inputs_text = processor(text=queries)

    # Forward pass
    with torch.no_grad():
        image_embeddings = model(**inputs_images).embeddings
        query_embeddings = model(**inputs_text).embeddings

    # Score the queries against the images
    scores = processor.score_retrieval(query_embeddings, image_embeddings)
    print(f"Retrieval scores (query x image): {scores}")


if __name__ == "__main__":
    main()
