import argparse
import os
import sys
from functools import partial
from unittest.mock import patch

import rebel
import torch
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS

GLOBALS.in_onnx_export = True

sys.path.append(os.path.join(sys.path[0], "yolov7"))
from yolov7.models.experimental import attempt_load


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov7",
        choices=["yolov7", "yolov7-tiny", "yolov7x"],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    torch.hub.load_state_dict_from_url(
        f"https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{model_name}.pt",
        ".",
        map_location=torch.device("cpu"),
    )

    # Override torch.load to set weights_only=False (default is True in PyTorch 2.6).
    with patch("torch.load", partial(torch.load, weights_only=False)):
        model = attempt_load(model_name + ".pt", map_location=torch.device("cpu"))
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
