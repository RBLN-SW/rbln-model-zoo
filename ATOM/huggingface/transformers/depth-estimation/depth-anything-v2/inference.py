import argparse
import os

import requests
import torch
from optimum.rbln import RBLNAutoModelForDepthEstimation
from PIL import Image
from transformers import AutoImageProcessor


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["small", "base", "large"],
        default="small",
        help="(str) type, Size of Model. [small, base, large]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"depth-anything/Depth-Anything-V2-{args.model_name.title()}-hf"

    # Load compiled model
    model = RBLNAutoModelForDepthEstimation.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,  # export a PyTorch model to RBLN model with optimum
    )

    # Load the image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Prepare image for the model
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(images=image, return_tensors="pt")

    # Inference
    predicted_depth = model(**inputs).predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze()
    formatted = (output * 255 / torch.max(output)).numpy().astype("uint8")
    depth_image = Image.fromarray(formatted)
    depth_image.save("depth_image.png")


if __name__ == "__main__":
    main()
