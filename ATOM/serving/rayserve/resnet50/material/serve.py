# File name: resnet50.py
import io
import json
import os

import ray
import rebel
import torch
from PIL import Image
from ray import serve
from starlette.requests import Request
from torchvision.models import ResNet50_Weights

ray.init(resources={"RBLN": 1})


@ray.remote(resources={"RBLN": 1})
class RBLNActor:
    def getDeviceId(self):
        return ray.get_runtime_context().get_accelerator_ids()["RBLN"]


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4})
class Resnet50:
    def __init__(self, rbln_actor: RBLNActor):
        self.initialized = False
        self.weights = None
        self.rbln_actor = rbln_actor
        self.ids = ray.get(rbln_actor.getDeviceId.remote())
        self.rbln_devices()
        self.initialize()

    def initialize(self):
        """
        Initialize model. This will be called during model loading time
        :return:
        """
        model_path = "./resnet50.rbln"
        if not os.path.isfile(model_path):
            raise RuntimeError(
                f"[RBLN ERROR] File not found at the specified model_path({model_path})."
            )
        self.module = rebel.Runtime(
            model_path, tensor_type="pt", device=int(self.ids[0])
        )
        self.weights = ResNet50_Weights.DEFAULT
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
            image = Image.open(io.BytesIO(input_data))
        except Exception as e:
            raise ValueError(f"[RBLN][ERROR]Invalid image data: {e}") from e
        prep = self.weights.transforms()
        batch = prep(image).unsqueeze(0)
        preprocessed_data = batch.numpy()

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
        score, class_id = torch.topk(inference_output, 1, dim=1)
        category_name = self.weights.meta["categories"][class_id]
        return category_name

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
        category_name = self.postprocess(model_output)

        return json.dumps({"result": category_name})

    async def __call__(self, http_request: Request) -> str:
        image_byte = await http_request.body()
        return self.handle(image_byte)


rbln_actor = RBLNActor.remote()
app = Resnet50.bind(rbln_actor)
