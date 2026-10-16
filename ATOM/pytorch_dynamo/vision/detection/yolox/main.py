import argparse
import os
import sys
import urllib.request
from typing import List, Tuple

import cv2
import numpy as np
import rebel  # noqa: F401  # needed to use torch dynamo's "rbln" backend.
import torch

sys.path.insert(0, os.path.join(sys.path[0], "YOLOX"))
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import postprocess as postprocessor
from yolox.utils import vis


def preprocess(
    img: np.array, test_size: List = [640, 640]
) -> Tuple[torch.tensor, dict]:
    ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
    img_info = {
        "height": img.shape[0],
        "width": img.shape[1],
        "ratio": ratio,
        "raw_img": img,
    }
    preprocessor = ValTransform(legacy=False)
    img, _ = preprocessor(img, None, input_size=test_size)
    img = torch.from_numpy(img).unsqueeze(0).float().to("cpu")

    return img, img_info


def visualization(
    outputs: torch.tensor, img_info: dict, class_names: tuple, result_path: str
):
    if outputs[0] is None:  # no detection result
        cv2.imwrite(result_path, img_info["raw_img"].copy())
    else:
        outputs = outputs[0].cpu()
        bboxes = outputs[:, 0:4]
        bboxes /= img_info["ratio"]  # preprocessing: resize
        cls = outputs[:, 6]
        scores = outputs[:, 4] * outputs[:, 5]
        vis_img = vis(
            img_info["raw_img"].copy(),
            bboxes,
            scores,
            cls,
            conf=0.35,
            class_names=class_names,
        )
        cv2.imwrite(result_path, vis_img)


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolox_s",
        choices=[
            "yolox_nano",
            "yolox_tiny",
            "yolox_s",
            "yolox_m",
            "yolox_l",
            "yolox_x",
            "yolov3",
        ],  # yolov3 is YOLOX-Darknet53
        help="available model variations",
    )

    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name
    input_size = 416 if "nano" in model_name or "tiny" in model_name else 640

    # Prepare input image
    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    img_path = "./tabby.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())
    img = cv2.imread(img_path)
    batch, img_info = preprocess(img, test_size=[input_size, input_size])
    batch = np.ascontiguousarray(batch.numpy(), dtype=np.float32)
    batch_tensor = torch.from_numpy(batch)

    # Load model
    model = torch.hub.load("Megvii-BaseDetection/YOLOX", model_name, pretrained=True)
    model.eval()

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

    # Get result image
    rebel_post_output = postprocessor(
        torch.tensor(result).to("cpu"),
        len(COCO_CLASSES),
        conf_thre=0.25,
        nms_thre=0.45,
        class_agnostic=True,
    )
    visualization(
        rebel_post_output, img_info, COCO_CLASSES, result_path=f"tabby_{model_name}.jpg"
    )


if __name__ == "__main__":
    main()
