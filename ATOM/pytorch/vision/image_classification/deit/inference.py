import argparse
import urllib.request

import rebel  # RBLN Runtime
import torch
import torchvision.transforms as transforms
from torchvision.io.image import read_image

_CHANNEL_MEANS = [123.68, 116.28, 103.53]
_CHANNEL_STDS = [58.395, 57.120, 57.385]


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="base",
        choices=[
            "tiny",
            "small",
            "base",
            "tiny_distilled",
            "small_distilled",
            "base_distilled",
            "base_384",
            "base_distilled_384",
        ],
        help="(str) type, DeiT model name. (tiny, small, base, tiny_distilled,"
        " small_distilled, base_distilled, base_384, base_distilled_384)",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name
    resize_size = 384 if "384" in model_name else 256
    input_size = 384 if "384" in model_name else 224

    # Prepare input image
    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    img_path = "./tabby.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())
    img = read_image(img_path)

    # Preprocessing Image
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(input_size),
            transforms.Normalize(_CHANNEL_MEANS, _CHANNEL_STDS),
        ]
    )
    batch = preprocess(img.to(torch.float32)).unsqueeze(0)

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"{model_name}.rbln")

    # Run inference
    rebel_result = module.run(batch.numpy())

    # Display results
    class_idx = torch.argmax(torch.tensor(rebel_result)).item()
    print(f"Top1 Classification Index: {class_idx}")  # 281: "tabby, tabby cat"


if __name__ == "__main__":
    main()
