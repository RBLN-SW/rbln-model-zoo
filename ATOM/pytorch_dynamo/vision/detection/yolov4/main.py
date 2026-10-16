import argparse
import os
import random
import sys
import urllib.request

import cv2
import numpy as np
import rebel  # noqa: F401  # needed to use torch dynamo's "rbln" backend.
import torch

sys.path.append(os.path.join(sys.path[0], "yolov4"))
from yolov4.models.models import Darknet, load_darknet_weights
from yolov4.utils.datasets import letterbox
from yolov4.utils.general import non_max_suppression, scale_coords
from yolov4.utils.plots import plot_one_box


def preprocess(image):
    preprocess_input = letterbox(image, (640, 640), auto=False)[0]
    preprocess_input = preprocess_input[:, :, ::-1].transpose(2, 0, 1)
    preprocess_input = np.ascontiguousarray(preprocess_input, dtype=np.float32)
    preprocess_input = preprocess_input[None]
    preprocess_input /= 255

    return preprocess_input


def postprocess(outputs, input_image, origin_image):
    pred = non_max_suppression(outputs, 0.4, 0.5)[0]
    pred[:, :4] = scale_coords(
        input_image.shape[2:], pred[:, :4], origin_image.shape
    ).round()
    names_path = os.path.abspath(os.path.dirname(__file__)) + "/yolov4/data/coco.names"
    with open(names_path) as f:
        names = f.read().split("\n")
        names_list = list(filter(None, names))
    colors = [
        [random.randint(0, 255) for _ in range(3)] for _ in range(len(names_list))
    ]
    for *xyxy, conf, cls in pred:
        label = "%s %.2f" % (names_list[int(cls)], conf)
        plot_one_box(
            xyxy, origin_image, label=label, color=colors[int(cls)], line_thickness=3
        )


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov4",
        choices=["yolov4", "yolov4-csp-s-mish", "yolov4-csp-x-mish"],
        help="available model variations",
    )
    parser.add_argument("--all", type=bool, default=False)
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
    batch_tensor = torch.from_numpy(batch)

    # Load model
    weight = model_name + ".weights"
    torch.hub.download_url_to_file(
        f"https://github.com/AlexeyAB/darknet/releases/download/yolov4/{weight}", weight
    )
    cfg = (
        os.path.abspath(os.path.dirname(__file__))
        + "/yolov4/cfg/"
        + model_name
        + ".cfg"
    )
    model = Darknet(cfg, weight)
    load_darknet_weights(model, weight)
    model.eval()
    # Pre-tracing before torch.compile
    model(batch_tensor)

    # Compile the model
    torch.compiler.reset()
    torch_module = torch.compile(
        model,
        backend="rbln",
        # Disable dynamic shape support, as the RBLN backend currently does not support it
        dynamic=False,
        options={"cache_dir": f"./{model_name}_cache"},
    )
    # (Optional) First call of forward invokes the compilation
    torch_module(batch_tensor)
    # After that, You can use models compiled for RBLN hardware
    result = torch_module(batch_tensor)

    # Save result
    postprocess(result[0], batch, img)
    cv2.imwrite(f"people_{model_name}.jpg", img)


if __name__ == "__main__":
    main()
