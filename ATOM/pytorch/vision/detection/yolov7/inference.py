import argparse
import os
import random
import sys
import urllib.request

import cv2
import numpy as np
import rebel
import torch
import yaml

sys.path.append(os.path.join(sys.path[0], "yolov7"))
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression as nms
from yolov7.utils.general import scale_coords
from yolov7.utils.plots import plot_one_box


def preprocess(image):
    preprocess_input = letterbox(image, (640, 640), 32, auto=False)[0]
    preprocess_input = preprocess_input[:, :, ::-1].transpose(2, 0, 1)
    preprocess_input = np.ascontiguousarray(preprocess_input, dtype=np.float32)
    preprocess_input = preprocess_input[None]
    preprocess_input /= 255

    return preprocess_input


def postprocess(outputs, input_image, origin_image):
    pred = nms(torch.from_numpy(outputs[0]), 0.25, 0.45, None, False)[0]
    pred[:, :4] = scale_coords(
        input_image.shape[2:], pred[:, :4], origin_image.shape
    ).round()
    yaml_path = os.path.abspath(os.path.dirname(__file__)) + "/yolov7/data/coco.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = list(data["names"])
    colors = [random.randint(0, 255)]
    for *xyxy, conf, cls in reversed(pred):
        c = int(cls)
        label = f"{names[c]} {conf:.2f}"
        plot_one_box(xyxy, origin_image, label=label, color=colors[0], line_thickness=3)


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov7",
        choices=["yolov7", "yolov7-tiny", "yolov7x"],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    # Prepare input image
    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/people4.jpg"
    img_path = "./people.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())
    img = cv2.imread(img_path)
    batch = preprocess(img)

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"{model_name}.rbln")

    rebel_result = module.run(batch)

    postprocess(rebel_result, batch, img)
    cv2.imwrite(f"people_{model_name}.jpg", img)


if __name__ == "__main__":
    main()
