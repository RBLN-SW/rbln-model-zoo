import argparse
import os
import sys
from unittest.mock import patch

import rebel
import torch

sys.path.append(os.path.join(sys.path[0], "ultralytics"))
from ultralytics import YOLO
from ultralytics.nn.modules.head import Pose


def _kpts_decode_traceable(self, bs, kpts):
    nkpt, ndim = self.kpt_shape
    n = kpts.shape[-1]
    y = kpts.view(bs, nkpt, ndim, n)
    x = (y[:, :, 0, :] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
    yc = (y[:, :, 1, :] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
    if ndim == 3:
        v = torch.sigmoid(y[:, :, 2, :])
        out = torch.stack([x, yc, v], dim=2)
    else:
        out = torch.stack([x, yc], dim=2)
    return out.view(bs, nkpt * ndim, n)


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

    model = YOLO(model_name + ".pt").model.eval()

    input_info = [
        ("input_np", [1, 3, 640, 640], torch.float32),
    ]
    with patch.object(Pose, "kpts_decode", _kpts_decode_traceable):
        compiled_model = rebel.compile_from_torch(model, input_info)
    compiled_model.save(f"{model_name}.rbln")


if __name__ == "__main__":
    main()
