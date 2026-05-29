import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np
import rebel
import torch

sys.path.append(os.path.join(sys.path[0], "ultralytics"))
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression as nms
from ultralytics.utils.ops import scale_boxes, scale_coords
from ultralytics.utils.plotting import Annotator


def preprocess(image):
    x = LetterBox(new_shape=(640, 640))(image=image)
    x = x.transpose((2, 0, 1))[::-1]
    x = np.ascontiguousarray(x, dtype=np.float32)[None]
    x /= 255
    return x


def postprocess(outputs, input_image, origin_image):
    pred = nms(torch.from_numpy(outputs), 0.25, 0.7, False, max_det=300, nc=1)[0]
    pred[:, :4] = scale_boxes(input_image.shape[2:], pred[:, :4], origin_image.shape)
    pred_kpts = pred[:, 6:].view(len(pred), 17, 3) if len(pred) else pred[:, 6:]
    pred_kpts = scale_coords(input_image.shape[2:], pred_kpts, origin_image.shape)

    annotator = Annotator(origin_image, line_width=3)
    for *xyxy, conf, _ in reversed(pred[:, :6]):
        annotator.box_label(xyxy, label=f"people {conf:.2f}")
    for k in reversed(pred_kpts):
        annotator.kpts(k, shape=origin_image.shape[:2], radius=5, kpt_line=True)

    return annotator.result()


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolo11s-pose",
        choices=[
            "yolo11s-pose",
            "yolo11n-pose",
            "yolo11m-pose",
            "yolo11l-pose",
            "yolo11x-pose",
        ],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    img_url = "https://ultralytics.com/images/bus.jpg"
    img_path = "./bus.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())
    img = cv2.imread(img_path)
    batch = preprocess(img)

    module = rebel.Runtime(f"{model_name}.rbln")
    rebel_result = module.run(batch)
    rebel_post_output = postprocess(rebel_result[0], batch, img)
    cv2.imwrite(f"bus_{model_name}.jpg", rebel_post_output)


if __name__ == "__main__":
    main()
