import argparse

import rebel  # RBLN Compiler
import torch
import torchvision


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=[
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_b5",
            "efficientnet_b6",
            "efficientnet_b7",
            "wide_resnet50_2",
            "wide_resnet101_2",
            "mnasnet0_5",
            "mnasnet0_75",
            "mnasnet1_0",
            "mnasnet1_3",
            "mobilenet_v2",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "alexnet",
            "vgg11",
            "vgg11_bn",
            "vgg13",
            "vgg13_bn",
            "vgg16",
            "vgg16_bn",
            "vgg19",
            "vgg19_bn",
            "squeezenet1_0",
            "squeezenet1_1",
            "shufflenet_v2_x0_5",
            "shufflenet_v2_x1_0",
            "shufflenet_v2_x1_5",
            "shufflenet_v2_x2_0",
            "densenet121",
            "densenet161",
            "densenet169",
            "densenet201",
            "googlenet",
            "inception_v3",
            "regnet_y_400mf",
            "regnet_y_800mf",
            "regnet_y_1_6gf",
            "regnet_y_3_2gf",
            "regnet_y_8gf",
            "regnet_y_16gf",
            "regnet_y_32gf",
            "regnet_y_128gf",
            "regnet_x_400mf",
            "regnet_x_800mf",
            "regnet_x_1_6gf",
            "regnet_x_3_2gf",
            "regnet_x_8gf",
            "regnet_x_16gf",
            "regnet_x_32gf",
            "efficientnet_v2_s",
            "efficientnet_v2_m",
            "efficientnet_v2_l",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "resnext101_64x4d",
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large",
        ],
        help="(str) type, torchvision model name.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    # Instantiate TorchVision model
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    model = getattr(torchvision.models, model_name)(weights=weights).eval()

    # Get input shape
    input_np = torch.rand(1, 3, 224, 224)
    input_np = weights.transforms()(input_np)

    # Compile torch model for ATOM
    input_info = [
        ("input_np", list(input_np.shape), torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"{model_name}.rbln")


if __name__ == "__main__":
    main()
