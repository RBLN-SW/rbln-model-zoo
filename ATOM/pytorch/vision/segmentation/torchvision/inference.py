import argparse
import urllib.request

import rebel
import torch
from PIL import Image
from torchvision import transforms


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

    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    img_path = "./tabby.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())
        img = Image.open(img_path)
        resize_image = img.resize((640, 640))
        resize_image = resize_image.convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_data = preprocess(resize_image).unsqueeze(0)

    module = rebel.Runtime(f"./{args.model_name}.rbln", tensor_type="pt")
    out = module(input_data)[0]
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    save_img = Image.fromarray(out.argmax(0).byte().numpy()).resize(
        img.size, resample=0
    )
    save_img.putpalette(colors)
    save_img.save("REBEL.png")


if __name__ == "__main__":
    main()
