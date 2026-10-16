# Ultralytics YOLO 🚀, AGPL-3.0 license
import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np
import rebel
import torch
import yaml

sys.path.append(os.path.join(sys.path[0], "ultralytics"))
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression as nms
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.plotting import Annotator


def preprocess(image):
    preprocess_input = LetterBox(new_shape=(640, 640))(image=image)
    preprocess_input = preprocess_input.transpose((2, 0, 1))[::-1]
    preprocess_input = np.ascontiguousarray(preprocess_input, dtype=np.float32)
    preprocess_input = preprocess_input[None]
    preprocess_input /= 255

    return preprocess_input


# https://github.com/ultralytics/ultralytics/blob/6dcc4a0610bf445212253fb51b24e29429a2bcc3/ultralytics/nn/modules/head.py#L133
def postprocess_in_detect(preds: torch.Tensor, max_det: int = 300, nc: int = 80):
    batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
    boxes, scores = preds.split([4, nc], dim=-1)
    index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
    boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
    scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
    scores, index = scores.flatten(1).topk(min(max_det, anchors))
    i = torch.arange(batch_size)[..., None]  # batch indices
    return torch.cat(
        [boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()],
        dim=-1,
    )


def postprocess(outputs, input_image, origin_image):
    pred = postprocess_in_detect(torch.from_numpy(outputs).permute(0, 2, 1))
    pred = nms(pred, 0.25, 0.45, None, False, max_det=1000)[0]
    pred[:, :4] = scale_boxes(input_image.shape[2:], pred[:, :4], origin_image.shape)
    annotator = Annotator(origin_image, line_width=3)
    yaml_path = (
        os.path.abspath(os.path.dirname(__file__))
        + "/ultralytics/ultralytics/cfg/datasets/coco128.yaml"
    )
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = list(data["names"].values())
    for *xyxy, conf, cls in reversed(pred):
        c = int(cls)
        label = f"{names[c]} {conf:.2f}"
        annotator.box_label(xyxy, label=label)

    return annotator.result()


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov10s",
        choices=[
            "yolov10s",
            "yolov10n",
            "yolov10b",
            "yolov10m",
            "yolov10l",
            "yolov10x",
        ],
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

    rbln_result = module.run(batch)

    rbln_post_output = postprocess(rbln_result[0], batch, img)
    cv2.imwrite(f"people_{model_name}.jpg", rbln_post_output)


if __name__ == "__main__":
    main()
