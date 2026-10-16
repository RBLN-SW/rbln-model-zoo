import argparse
import os

from optimum.rbln import RBLNAutoModelForDepthEstimation


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["small", "base", "large"],
        default="small",
        help="(str) type, Size of Model. [small, base, large]",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=518,
        help="(int) type, height of the image.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=686,
        help="(int) type, width of the image.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"depth-anything/Depth-Anything-V2-{args.model_name.title()}-hf"

    # Compile and export
    model = RBLNAutoModelForDepthEstimation.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_image_size=(args.height, args.width),
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
