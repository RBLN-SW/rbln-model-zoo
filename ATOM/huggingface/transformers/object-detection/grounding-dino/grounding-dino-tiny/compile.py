import os

from optimum.rbln import RBLNAutoModelForZeroShotObjectDetection


def main():
    model_id = "IDEA-Research/grounding-dino-tiny"

    # Compile and export
    model = RBLNAutoModelForZeroShotObjectDetection.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_config={
            "backbone": {
                "image_size": (1333, 1333),
                "batch_size": 1,
            },
            "encoder": {
                "image_size": (1333, 1333),
                "batch_size": 1,
            },
            "decoder": {
                "image_size": (1333, 1333),
                "batch_size": 1,
            },
            "text_backbone": {
                "batch_size": 1,
            },
        },
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
