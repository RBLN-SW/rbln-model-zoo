# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import urllib.request

import cv2
import fire
import numpy as np
import torch
import tritonclient.http as httpclient
import yaml
from tritonclient.utils import np_to_triton_dtype
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression as nms
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.plotting import Annotator

DEFAULT_URL = "localhost:8000"
DEFAULT_REQUESTS = 10
MODEL_NAME = "yolov8l"


def preprocess(image):
    preprocess_input = LetterBox(new_shape=(640, 640))(image=image)
    preprocess_input = preprocess_input.transpose((2, 0, 1))[::-1]
    preprocess_input = np.ascontiguousarray(preprocess_input, dtype=np.float32)
    preprocess_input = preprocess_input[None]
    preprocess_input /= 255

    return preprocess_input


def postprocess(outputs, input_image, origin_image):
    pred = nms(torch.from_numpy(outputs), 0.25, 0.45, None, False, max_det=1000)[0]
    pred[:, :4] = scale_boxes(input_image.shape[2:], pred[:, :4], origin_image.shape)
    annotator = Annotator(origin_image, line_width=3)
    yaml_path = os.path.abspath(os.path.dirname(__file__)) + "/coco128.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = list(data["names"].values())
    for *xyxy, conf, cls in reversed(pred):
        c = int(cls)
        label = f"{names[c]} {conf:.2f}"
        annotator.box_label(xyxy, label=label)

    return annotator.result()


def infer(
    url: str = DEFAULT_URL,
    requests: int = DEFAULT_REQUESTS,
    verbose: bool = False,
):
    # Prepare input image
    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/people4.jpg"
    img_path = "./people.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())
    img = cv2.imread(img_path)
    batch = preprocess(img)

    # configure httpclient
    with httpclient.InferenceServerClient(url=url, verbose=verbose) as client:
        inputs = []
        inputs.append(
            httpclient.InferInput(
                "INPUT__0", batch.shape, np_to_triton_dtype(batch.dtype)
            )
        )
        inputs[0].set_data_from_numpy(batch)
        outputs = [
            httpclient.InferRequestedOutput("OUTPUT__0"),
        ]
        responses = []
        # inference
        for i in range(requests):
            responses.append(
                client.infer(MODEL_NAME, inputs, request_id=str(i), outputs=outputs)
            )
        # check result
        for i, response in enumerate(responses):
            rebel_result = response.as_numpy("OUTPUT__0")
            rebel_post_output = postprocess(rebel_result, batch, img)
            ret = cv2.imwrite(f"people_{MODEL_NAME}_{i}.jpg", rebel_post_output)
            assert ret is not None


if __name__ == "__main__":
    fire.Fire(infer)
