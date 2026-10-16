import argparse
import os

from optimum.rbln import RBLNAutoModelForDepthEstimation


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_size",
        type=int,
        default=384,
        help="(int) type, Size of the image.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "Intel/dpt-large"

    # Compile and export
    model = RBLNAutoModelForDepthEstimation.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_image_size=args.image_size,  # target image size of the model
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
