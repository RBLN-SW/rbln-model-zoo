import argparse

import rebel
import torch
import torchvision

model_weights_map = {
    "fcn_resnet50": "FCN_ResNet50_Weights",
    "fcn_resnet101": "FCN_ResNet101_Weights",
    "deeplabv3_resnet50": "DeepLabV3_ResNet50_Weights",
    "deeplabv3_resnet101": "DeepLabV3_ResNet101_Weights",
    "deeplabv3_mobilenet_v3_large": "DeepLabV3_MobileNet_V3_Large_Weights",
}


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inp):
        result = self.model(*inp)
        return result["out"]


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "fcn_resnet50",
            "fcn_resnet101",
            "deeplabv3_resnet50",
            "deeplabv3_resnet101",
            "deeplabv3_mobilenet_v3_large",
        ],
        default="fcn_resnet50",
        help="(str) type, torchvision semantic segmentation model name.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()

    model_weights = model_weights_map.get(args.model_name)
    weights = getattr(torchvision.models.segmentation, model_weights).DEFAULT
    model = getattr(torchvision.models.segmentation, args.model_name)(weights=weights)
    model.eval()
    input_ = torch.rand(1, 3, 640, 640)
    input_info = [
        ("input_img", list(input_.shape), torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(TraceWrapper(model), input_info)

    compiled_model.save(f"./{args.model_name}.rbln")


if __name__ == "__main__":
    main()
