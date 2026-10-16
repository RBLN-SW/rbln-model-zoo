# Ultralytics YOLO 🚀, AGPL-3.0 license
import argparse
import os
import sys

import rebel
import torch

sys.path.append(os.path.join(sys.path[0], "ultralytics"))
from ultralytics import YOLO


class YOLOv10Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(YOLOv10Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        y = []  # outputs
        for m in self.model.model[:-1]:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.model.save else None)  # save output

        fm = self.model.model[-1]  # v10Detect
        if fm.f != -1:  # if not from previous layer
            x = (
                y[fm.f]
                if isinstance(fm.f, int)
                else [x if j == -1 else y[j] for j in fm.f]
            )  # from earlier layers

        # inference v10Detect.forward_end2end without v10Detect.postprocess
        # https://github.com/ultralytics/ultralytics/blob/6dcc4a0610bf445212253fb51b24e29429a2bcc3/ultralytics/nn/modules/head.py#L63
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat(
                (fm.one2one_cv2[i](x_detach[i]), fm.one2one_cv3[i](x_detach[i])), 1
            )
            for i in range(fm.nl)
        ]
        for i in range(fm.nl):
            x[i] = torch.cat((fm.cv2[i](x[i]), fm.cv3[i](x[i])), 1)
        out = fm._inference(one2one)  # run
        return [out] + x + one2one


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov10s",
        choices=[
            "yolov10s",
            "yolov10n",
            "yolov10m",
            "yolov10b",
            "yolov10l",
            "yolov10x",
        ],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    model = YOLO(model_name + ".pt").model
    model.eval()
    model = YOLOv10Wrapper(model)

    # Compile torch model for ATOM
    input_info = [
        ("input_np", [1, 3, 640, 640], torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"{model_name}.rbln")


if __name__ == "__main__":
    main()
