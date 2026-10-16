import os

from optimum.rbln import RBLNAutoModelForImageClassification


def main():
    model_id = "microsoft/resnet-50"

    # Compile and export
    model = RBLNAutoModelForImageClassification.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_image_size=224,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
