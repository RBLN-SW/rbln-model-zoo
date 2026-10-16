import argparse

import rebel  # RBLN Compiler
import torch


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolox_s",
        choices=[
            "yolox_nano",
            "yolox_tiny",
            "yolox_s",
            "yolox_m",
            "yolox_l",
            "yolox_x",
            "yolov3",
        ],  # yolov3 is YOLOX-Darknet53
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name
    input_size = 416 if "nano" in model_name or "tiny" in model_name else 640

    # Instantiate TorchVision model
    model = torch.hub.load("Megvii-BaseDetection/YOLOX", model_name, pretrained=True)
    model.eval()

    # Compile torch model for ATOM
    input_info = [
        ("input_np", [1, 3, input_size, input_size], torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"{model_name}.rbln")


if __name__ == "__main__":
    main()
