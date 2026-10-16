from argparse import ArgumentParser

import rebel
import torch
from ultralytics import YOLO


def parsing_argument() -> object:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name",
        "--model_name",
        dest="model_name",
        default="yolov8n",
        choices=[
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
            "yolov8x",
        ],
        help="YOLO model name",
    )
    return parser.parse_args()


def main() -> None:
    args = parsing_argument()

    yolo = YOLO(f"{args.model_name}.pt")
    model = yolo.model.eval()

    # Warm up to build the computation graph.
    model(torch.zeros(1, 3, 640, 640))

    compiled_model = rebel.compile_from_torch(
        model, [("x", [1, 3, 640, 640], "float32")]
    )
    compiled_model.save(f"{args.model_name}.rbln")


if __name__ == "__main__":
    main()
