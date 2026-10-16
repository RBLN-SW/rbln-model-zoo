import os

from optimum.rbln import RBLNAutoModelForSequenceClassification


def main():
    model_id = "Dongjin-kr/ko-reranker"

    # Compile and export
    model = RBLNAutoModelForSequenceClassification.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=512,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
