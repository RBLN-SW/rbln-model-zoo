import os

import torch
from huggingface_hub import hf_hub_download
from optimum.rbln import RBLNTimeSeriesTransformerForPrediction


def main():
    model_id = "huggingface/time-series-transformer-tourism-monthly"
    model_path = os.path.basename(model_id)

    file = hf_hub_download(
        repo_id="hf-internal-testing/tourism-monthly-batch",
        filename="train-batch.pt",
        repo_type="dataset",
    )
    batch = torch.load(file)  # batch_size 64

    # Compile and export
    model = RBLNTimeSeriesTransformerForPrediction.from_pretrained(
        model_path,
        export=False,
    )

    outputs = model.generate(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        static_categorical_features=batch["static_categorical_features"],
        static_real_features=batch["static_real_features"],
        future_time_features=batch["future_time_features"],
    )

    mean_prediction = outputs.sequences.mean(dim=1)
    print(f"Result: {mean_prediction}")


if __name__ == "__main__":
    main()
