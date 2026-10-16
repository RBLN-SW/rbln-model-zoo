# File name: yolov8.py
import io
import json
import os

import numpy as np
import ray
import rebel
import torch
import yaml
from PIL import Image
from ray import serve
from starlette.requests import Request
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression as nms
from ultralytics.utils.ops import scale_boxes

ray.init(resources={"RBLN": 1})


@ray.remote(resources={"RBLN": 1})
class RBLNActor:
    def getDeviceId(self):
        return ray.get_runtime_context().get_accelerator_ids()["RBLN"]


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 16})
class Yolov8:
    def __init__(self, rbln_actor: RBLNActor):
        self.initialized = False
        self.input_image = None
        self.rbln_actor = rbln_actor
        self.ids = ray.get(rbln_actor.getDeviceId.remote())
        self.rbln_devices()
        self.initialize()

    def initialize(self):
        """
        Initialize model. This will be called during model loading time
        :return:
        """
        model_path = "./yolov8l.rbln"
        if not os.path.isfile(model_path):
            raise RuntimeError(
                f"[RBLN ERROR] File not found at the specified model_path({model_path})."
            )
        self.module = rebel.Runtime(
            model_path, tensor_type="pt", device=int(self.ids[0])
        )
        self.initialized = True

    def rbln_devices(self):
        """
        Redefine the environment variables to be passed to the RBLN runtime
        :return:
        """
        if self.ids is None or len(self.ids) <= 0:
            os.environ.pop("RBLN_DEVICES")
        os.environ["RBLN_DEVICES"] = ",".join(self.ids)

    def preprocess(self, input_data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        assert input_data is not None, print(
            "[RBLN][ERROR] Data not found with client request."
        )
        if not isinstance(input_data, (bytes, bytearray)):
            raise ValueError("[RBLN][ERROR] Preprocessed data is not binary data.")

        try:
            image = Image.open(io.BytesIO(input_data)).convert("RGB")
        except Exception as e:
            raise ValueError(f"[RBLN][ERROR]Invalid image data: {e}") from e

        image = np.array(image)

        preprocessed_data = LetterBox(new_shape=(640, 640))(image=image)
        preprocessed_data = preprocessed_data.transpose((2, 0, 1))[::-1]
        preprocessed_data = np.ascontiguousarray(preprocessed_data, dtype=np.float32)
        preprocessed_data = preprocessed_data[None]
        preprocessed_data /= 255
        self.input_image = preprocessed_data

        return torch.from_numpy(preprocessed_data)

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """

        model_output = self.module.run(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format

        pred = nms(inference_output, 0.25, 0.45, None, False, max_det=1000)[0]
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

    def handle(self, data):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        result = self.postprocess(model_output)

        return json.dumps({"result": result})

    async def __call__(self, http_request: Request) -> str:
        image_byte = await http_request.body()
        return self.handle(image_byte)


rbln_actor = RBLNActor.remote()
app = Yolov8.bind(rbln_actor)
