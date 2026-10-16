import argparse
import os
import sys

import rebel
import torch

sys.path.append(os.path.join(sys.path[0], "yolov4"))
from yolov4.models.models import Darknet, load_darknet_weights


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov4",
        choices=["yolov4", "yolov4-csp-s-mish", "yolov4-csp-x-mish"],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    weight = model_name + ".weights"
    torch.hub.download_url_to_file(
        f"https://github.com/AlexeyAB/darknet/releases/download/yolov4/{weight}", weight
    )

    cfg = (
        os.path.abspath(os.path.dirname(__file__))
        + "/yolov4/cfg/"
        + model_name
        + ".cfg"
    )

    model = Darknet(cfg, weight)
    load_darknet_weights(model, weight)
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
