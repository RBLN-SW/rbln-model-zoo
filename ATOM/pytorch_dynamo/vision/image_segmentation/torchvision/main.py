import argparse
import urllib.request

import rebel  # noqa: F401  # needed to use torch dynamo's "rbln" backend
import torch
import torchvision
from PIL import Image
from torchvision import transforms

model_weights_map = {
    "fcn_resnet50": "FCN_ResNet50_Weights",
    "fcn_resnet101": "FCN_ResNet101_Weights",
    "deeplabv3_resnet50": "DeepLabV3_ResNet50_Weights",
    "deeplabv3_resnet101": "DeepLabV3_ResNet101_Weights",
    "deeplabv3_mobilenet_v3_large": "DeepLabV3_MobileNet_V3_Large_Weights",
}


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

    # Load and preprocess the image
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_data = preprocess(resize_image).unsqueeze(0)

    # Load the model
    model_weights = model_weights_map.get(args.model_name)
    weights = getattr(torchvision.models.segmentation, model_weights).DEFAULT
    model = getattr(torchvision.models.segmentation, args.model_name)(weights=weights)
    model.eval()

    # Compile the model using torch.compile with RBLN backend
    compiled_model = torch.compile(
        model,
        backend="rbln",
        # Disable dynamic shape support, as the RBLN backend currently does not support it
        dynamic=False,
        options={"cache_dir": f"./{args.model_name}"},
    )

    # (Optional) First call of forward invokes the compilation
    compiled_model(input_data)

    # Run inference using the compiled model
    out = compiled_model(input_data)["out"]

    # Post-process the output
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    save_img = Image.fromarray(out[0].argmax(0).byte().numpy()).resize(
        img.size, resample=0
    )
    save_img.putpalette(colors)
    save_img.save("REBEL.png")

    print(f"Segmentation completed using {args.model_name} with RBLN backend.")


if __name__ == "__main__":
    main()
