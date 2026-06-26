import argparse
import urllib.request

import rebel  # noqa: F401  # needed to use torch dynamo's "rbln" backend.
import torch
import torchvision
from torchvision.io.image import read_image


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

    # Prepare input image
    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    img_path = "./tabby.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())
    img = read_image(img_path)
    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0)

    # Compile the model
    model = torch.compile(
        model,
        backend="rbln",
        # Disable dynamic shape support, as the RBLN backend currently does not support it
        dynamic=False,
        options={"cache_dir": f"./{model_name}"},
    )

    # (Optional) First call of forward invokes the compilation
    model(batch)

    # After that, You can use models compiled for RBLN hardware
    rbln_result = model(batch)

    # Display results
    score, class_id = torch.topk(rbln_result, 1, dim=1)
    category_name = weights.meta["categories"][class_id]
    print(f"Top1 category: {category_name}")


if __name__ == "__main__":
    main()
