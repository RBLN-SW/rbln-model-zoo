import argparse
import os

import rebel
import torch


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov5s",
        choices=["yolov5s", "yolov5n", "yolov5m", "yolov5l", "yolov5x"],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    # Load from the vendored submodule; the remote repo tracks a moving master.
    yolov5_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov5")
    model = torch.hub.load(yolov5_dir, model_name, source="local")
    model.eval()

    # Compile torch model for ATOM
    input_info = [
        ("input_np", [1, 3, 640, 640], torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"{model_name}.rbln")


if __name__ == "__main__":
    main()
