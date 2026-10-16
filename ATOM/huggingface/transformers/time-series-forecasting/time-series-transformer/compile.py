import os

from optimum.rbln import RBLNTimeSeriesTransformerForPrediction


def main():
    model_id = "huggingface/time-series-transformer-tourism-monthly"

    # Compile and export
    model = RBLNTimeSeriesTransformerForPrediction.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=64,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
