import argparse
import contextlib
import os
import sys
import urllib.request
import warnings

import cv2
import numpy as np
import rebel  # noqa: F401  # needed to use torch dynamo's "rbln" backend.
import torch
import yaml

sys.path.insert(0, os.path.join(sys.path[0], "yolov5"))
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression as nms
from yolov5.utils.general import scale_boxes
from yolov5.utils.plots import Annotator, colors


def preprocess(image):
    preprocess_input = letterbox(image, (640, 640), 32, auto=False)[0]
    preprocess_input = preprocess_input.transpose((2, 0, 1))[::-1]
    preprocess_input = np.ascontiguousarray(preprocess_input, dtype=np.float32)
    preprocess_input = preprocess_input[None]
    preprocess_input /= 255

    return preprocess_input


def postprocess(outputs, input_image, origin_image):
    pred = nms(outputs, 0.25, 0.45, None, False, max_det=1000)[0]
    annotator = Annotator(origin_image, line_width=3)
    pred[:, :4] = scale_boxes(
        input_image.shape[2:], pred[:, :4], origin_image.shape
    ).round()
    yaml_path = os.path.abspath(os.path.dirname(__file__)) + "/yolov5/data/coco128.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = list(data["names"].values())
    for *xyxy, conf, cls in reversed(pred):
        c = int(cls)
        label = f"{names[c]} {conf:.2f}"
        annotator.box_label(xyxy, label, color=colors(c, True))

    return annotator.result()


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov5s",
        choices=["yolov5s", "yolov5n", "yolov5m", "yolov5l", "yolov5x"],
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

    # Load from the vendored submodule; the remote repo tracks a moving master.
    yolov5_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov5")
    model = torch.hub.load(yolov5_dir, model_name, source="local")
    model.eval()

    # Disable capturing warnings for torch.compile
    torch._dynamo.allow_in_graph(warnings.simplefilter)

    def forward_pre_hook(_module, _inputs):
        warnings.catch_warnings = contextlib.nullcontext

    model.register_forward_pre_hook(forward_pre_hook)

    # Compile the model
    torch_module = torch.compile(
        model,
        backend="rbln",
        # Disable dynamic shape support,
        # as the RBLN backend currently does not support it
        dynamic=False,
        options={"cache_dir": f"./{model_name}_cache"},
    )
    # (Optional) First call of forward invokes the compilation
    torch_module(batch_tensor)
    # After that, You can use models compiled for RBLN hardware
    result = torch_module(batch_tensor)

    rebel_post_output = postprocess(result, batch, img)
    cv2.imwrite(f"people_{model_name}.jpg", rebel_post_output)


if __name__ == "__main__":
    main()
