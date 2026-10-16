# yolov8l_handler.py

"""
ModelHandler defines a custom model handler.
"""

import io
import os

import numpy as np
import PIL.Image as Image
import rebel  # RBLN Runtime
import torch
import yaml
from ts.torch_handler.base_handler import BaseHandler
from ultralytics.data.augment import LetterBox
from ultralytics.yolo.utils.ops import non_max_suppression as nms
from ultralytics.yolo.utils.ops import scale_boxes


class YOLOv8_Handler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.input_image = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        #  load the model, refer 'custom handler class' above for details
        model_dir = context.system_properties.get("model_dir")
        serialized_file = context.manifest["model"].get("serializedFile")
        model_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_path):
            raise RuntimeError(
                f"[RBLN ERROR] File not found at the specified model_path({model_path})."
            )

        self.module = rebel.Runtime(model_path, tensor_type="pt")
        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")

            image = Image.open(io.BytesIO(preprocessed_data)).convert("RGB")
            image = np.array(image)

            preprocessed_data = LetterBox(new_shape=(640, 640))(image=image)
            preprocessed_data = preprocessed_data.transpose((2, 0, 1))[::-1]
            preprocessed_data = preprocessed_data[None]
            preprocessed_data = np.ascontiguousarray(
                preprocessed_data, dtype=np.float32
            )
            preprocessed_data /= 255
            self.input_image = preprocessed_data

        return torch.from_numpy(preprocessed_data)

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.module.run(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output

        pred = nms(postprocess_output, 0.25, 0.45, None, False, max_det=1000)[0]
        pred[:, :4] = scale_boxes(
            self.input_image.shape[2:], pred[:, :4], self.input_image.shape
        )
        yaml_path = "./coco128.yaml"

        postprocess_output = []
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        names = list(data["names"].values())
        for *xyxy, conf, cls in reversed(pred):
            xyxy_str = f"{xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}"
            postprocess_output.append(
                f"xyxy : {xyxy_str}, conf : {conf}, cls : {names[int(cls)]}"
            )

        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return [{"result": self.postprocess(model_output[0])}]
