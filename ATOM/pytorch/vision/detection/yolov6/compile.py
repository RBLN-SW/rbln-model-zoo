import argparse
import os
import sys

import rebel
import torch

sys.path.append(os.path.join(sys.path[0], "YOLOv6"))


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov6s",
        choices=["yolov6s", "yolov6n", "yolov6m", "yolov6l"],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    model = torch.hub.load_state_dict_from_url(
        f"https://github.com/meituan/YOLOv6/releases/download/0.4.0/{model_name}.pt",
        ".",
        map_location=torch.device("cpu"),
    )["model"].float()
    model.export = True
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
