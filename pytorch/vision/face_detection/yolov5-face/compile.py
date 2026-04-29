import argparse
import os
import sys
from functools import partial
from unittest.mock import patch

import gdown
import rebel
import torch

sys.path.append(os.path.join(sys.path[0], "yolov5-face"))
from models.experimental import attempt_load

pt_file_urls = {
    "yolov5n-0.5": "https://drive.google.com/uc?id=1XJ8w55Y9Po7Y5WP4X1Kg1a77ok2tL_KY",
    "yolov5n-face": "https://drive.google.com/uc?id=18oenL6tjFkdR1f5IgpYeQfDFqU4w3jEr",
    "yolov5s-face": "https://drive.google.com/uc?id=1zxaHeLDyID9YU4-hqK7KNepXIwbTkRIO",
    "yolov5m-face": "https://drive.google.com/uc?id=1Sx-KEGXSxvPMS35JhzQKeRBiqC98VDDI",
}


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov5m-face",
        choices=["yolov5n-0.5", "yolov5n-face", "yolov5s-face", "yolov5m-face"],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    # Get official pretrained weight from google drive
    gdown.download(url=pt_file_urls[model_name], output=f"{model_name}.pt")

    # Override torch.load to set weights_only=False (default is True in PyTorch 2.6).
    with patch("torch.load", partial(torch.load, weights_only=False)):
        model = attempt_load(f"{model_name}.pt", map_location="cpu")

    # Preprocess model for rbln
    delattr(model.model[-1], "anchor_grid")
    model.model[-1].anchor_grid = [torch.zeros(1)] * 3
    model.model[-1].export_cat = True
    model.eval()

    # Compile torch model for ATOM
    input_info = [
        ("input_np", [1, 3, 384, 640], torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"{model_name}.rbln")


if __name__ == "__main__":
    main()
