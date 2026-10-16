import argparse
import os
import sys

import cv2
import numpy as np
import rebel
import torch

sys.path.append(os.path.join(sys.path[0], "yolov5-face"))
from utils.datasets import letterbox
from utils.general import non_max_suppression_face, scale_coords, xyxy2xywh


# ref: https://github.com/deepcam-cn/yolov5-face/blob/152c688d551aefb973b7b589fb0691c93dab3564/test_widerface.py#L26
def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


# ref: https://github.com/deepcam-cn/yolov5-face/blob/152c688d551aefb973b7b589fb0691c93dab3564/test_widerface.py#L51
def show_results(img, xywh, conf, landmarks, class_num):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(
        img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA
    )

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(int(class_num)) + ": " + str(conf)[:5]
    cv2.putText(
        img,
        label,
        (x1, y1 - 2),
        0,
        tl / 3,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return img


def preprocess(image):
    preprocess_input = letterbox(image, (384, 640))[0]
    preprocess_input = preprocess_input[:, :, ::-1].transpose(2, 0, 1)
    preprocess_input = np.ascontiguousarray(preprocess_input, dtype=np.float32)
    preprocess_input = preprocess_input[None]
    preprocess_input /= 255

    return preprocess_input


def postprocess(outputs, input_image, origin_image):
    pred = non_max_suppression_face(torch.from_numpy(outputs), 0.02, 0.5)[0]
    gn = torch.tensor(origin_image.shape)[[1, 0, 1, 0]]
    gn_lks = torch.tensor(origin_image.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]
    pred[:, :4] = scale_coords(
        input_image.shape[2:], pred[:, :4], origin_image.shape
    ).round()
    pred[:, 5:15] = scale_coords_landmarks(
        input_image.shape[2:], pred[:, 5:15], origin_image.shape
    ).round()
    for j in range(pred.size()[0]):
        xywh = (xyxy2xywh(pred[j, :4].view(1, 4)) / gn).view(-1)
        conf = pred[j, 4].cpu().numpy()
        landmarks = (pred[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
        class_num = pred[j, 15].cpu().numpy()
        result = show_results(origin_image, xywh, conf, landmarks, class_num)

    return result


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov5m-face",
        choices=["yolov5n-0.5", "yolov5n-face", "yolov5s-face", "yolov5m-face"],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    # Prepare input image
    img = cv2.imread(sys.path[0] + "/yolov5-face/data/images/test.jpg")
    batch = preprocess(img)

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"{model_name}.rbln")

    rebel_result = module.run(batch)

    rebel_post_output = postprocess(rebel_result, batch, img)
    cv2.imwrite(f"face-detect-{model_name}.jpg", rebel_post_output)


if __name__ == "__main__":
    main()
