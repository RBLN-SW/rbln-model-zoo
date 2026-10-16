import argparse
import os
import sys

import rebel
import torch

sys.path.append(os.path.join(sys.path[0], "ultralytics"))
from ultralytics import YOLO


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov8s",
        choices=["yolov8s", "yolov8n", "yolov8m", "yolov8l", "yolov8x"],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    model = YOLO(model_name + ".pt").model
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
