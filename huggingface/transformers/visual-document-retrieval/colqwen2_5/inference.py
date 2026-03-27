import os
import sys

sys.path.insert(0, os.path.join(sys.path[0], "colpali"))

import torch
from colpali.colpali_engine.models import ColQwen2_5_Processor
from optimum.rbln import RBLNColQwen2ForRetrieval
from PIL import Image


def main():
    model_id = "Metric-AI/ColQwen2.5-3b-multilingual-v1.0"

    # Load compiled model
    model = RBLNColQwen2ForRetrieval.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )
    processor = ColQwen2_5_Processor.from_pretrained(
        model_id, max_num_visual_tokens=768
    )

    # Your inputs
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (64, 64), color="black"),
    ]
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the inputs
    batch_images = processor.process_images(images)
    batch_queries = processor.process_queries(queries)

    # Forward pass
    with torch.no_grad():
        image_embeddings = model(**batch_images).embeddings
        query_embeddings = model(**batch_queries).embeddings

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    print("--- score ---")
    print(scores)


if __name__ == "__main__":
    main()
