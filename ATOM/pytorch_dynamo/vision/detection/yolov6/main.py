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

sys.path.insert(0, os.path.join(sys.path[0], "YOLOv6"))
from YOLOv6.yolov6.core.inferer import Inferer
from YOLOv6.yolov6.data.data_augment import letterbox
from YOLOv6.yolov6.utils.nms import non_max_suppression as nms


def preprocess(image):
    preprocess_input = letterbox(image, (640, 640), stride=32, auto=False)[0]
    preprocess_input = preprocess_input.transpose((2, 0, 1))[::-1]
    preprocess_input = np.ascontiguousarray(preprocess_input, dtype=np.float32)
    preprocess_input = preprocess_input[None]
    preprocess_input /= 255

    return preprocess_input


def postprocess(outputs, input_image, origin_image):
    pred = nms(outputs, 0.5, 0.5, None, False, max_det=1000)[0]
    pred[:, :4] = Inferer.rescale(
        input_image.shape[2:], pred[:, :4], origin_image.shape
    ).round()
    yaml_path = os.path.abspath(os.path.dirname(__file__)) + "/YOLOv6/data/coco.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    class_names = data["names"]
    for *xyxy, conf, cls in reversed(pred):
        class_num = int(cls)
        label = f"{class_names[class_num]} {conf:.2f}"
        Inferer.plot_box_and_label(
            origin_image,
            max(round(sum(origin_image.shape) / 2 * 0.003), 2),
            xyxy,
            label,
            color=Inferer.generate_colors(class_num, True),
        )


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov6s",
        choices=["yolov6s", "yolov6n", "yolov6m", "yolov6l"],
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
    model = torch.hub.load_state_dict_from_url(
        f"https://github.com/meituan/YOLOv6/releases/download/0.4.0/{model_name}.pt",
        ".",
        map_location=torch.device("cpu"),
    )["model"].float()
    model.export = True
    model.eval()

    # Some calls on torch.Tensor in this model return a scalar type.

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

    postprocess(result, batch, img)
    cv2.imwrite(f"people_{model_name}.jpg", img)


if __name__ == "__main__":
    main()
