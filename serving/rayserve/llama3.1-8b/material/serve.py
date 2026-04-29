import json
import os
from unittest.mock import MagicMock

import ray
from fastapi import FastAPI, HTTPException
from ray import serve
from starlette.requests import Request
from starlette.responses import StreamingResponse
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.render.serving import OpenAIServingRender

from vllm import AsyncEngineArgs, AsyncLLMEngine

app = FastAPI()

ray.init(resources={"RBLN": 8})


@ray.remote(resources={"RBLN": 8})
class RBLNActor:
    def getDeviceId(self):
        return ray.get_runtime_context().get_accelerator_ids()["RBLN"]


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 16})
@serve.ingress(app)
class Llama3_1__8B:
    def __init__(self, rbln_actor: RBLNActor):
        """
        Initialize actor.
        :return:
        """
        self.engine = None
        self.rbln_actor = rbln_actor
        self.model_name = "Llama-3.1-8B-Instruct"
        self.raw_request = None
        self.vllm_engine = None
        self.openai_serving_models = None
        self.completion_service = None
        self.chat_completion_service = None
        self.ids = ray.get(rbln_actor.getDeviceId.remote())

        self.os_environment_vars()
        self.initialize()

    def os_environment_vars(self):
        """
        Redefine the environment variables to be passed to the RBLN runtime and vLLM
        :return:
        """
        if self.ids is None or len(self.ids) <= 0:
            os.environ.pop("RBLN_DEVICES")
        os.environ["RBLN_DEVICES"] = ",".join(self.ids)
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    def initialize(self):
        """
        Initialize model. This will be called during model loading time
        :return:
        """
        vllm_engine_args = AsyncEngineArgs(model=self.model_name)
        self.vllm_engine = AsyncLLMEngine.from_engine_args(vllm_engine_args)

        self.openai_serving_models = OpenAIServingModels(
            engine_client=self.vllm_engine,
            base_model_paths=[
                BaseModelPath(name=self.model_name, model_path=os.getcwd())
            ],
        )

        openai_serving_render = OpenAIServingRender(
            model_config=self.vllm_engine.model_config,
            renderer=self.vllm_engine.renderer,
            io_processor=self.vllm_engine.io_processor,
            model_registry=self.openai_serving_models.registry,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )

        self.completion_service = OpenAIServingCompletion(
            self.vllm_engine,
            self.openai_serving_models,
            openai_serving_render=openai_serving_render,
            request_logger=None,
        )

        self.chat_completion_service = OpenAIServingChat(
            self.vllm_engine,
            self.openai_serving_models,
            "assistant",
            openai_serving_render=openai_serving_render,
            request_logger=None,
            chat_template_content_format="auto",
            chat_template=None,
        )

        async def isd():
            return False

        self.raw_request = MagicMock()
        self.raw_request.headers = {}
        self.raw_request.is_disconnected = isd

    @app.post("/v1/chat/completions")
    async def chat_completion(self, http_request: Request):
        """
        Handle chat completion request.
        :param http_request: The HTTP request object
        :return: The chat completion response
        """
        try:
            json_string: dict = await http_request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format request")
        request: ChatCompletionRequest = ChatCompletionRequest.model_validate(
            json_string
        )

        g = await self.chat_completion_service.create_chat_completion(
            request, self.raw_request
        )

        if isinstance(g, ErrorResponse):
            return [g.model_dump()]

        if request.stream:

            async def stream_generator():
                async for response in g:
                    yield response

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            return [g.model_dump()]

    @app.post("/v1/completions")
    async def completion(self, http_request: Request):
        """
        Handle completion request.
        :param http_request: The HTTP request object
        :return: The completion response
        """
        try:
            json_string: dict = await http_request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format request")
        request: CompletionRequest = CompletionRequest.model_validate(json_string)

        g = await self.completion_service.create_completion(request, self.raw_request)

        if isinstance(g, ErrorResponse):
            return [g.model_dump()]

        if request.stream:

            async def stream_generator():
                async for response in g:
                    yield response

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            return [g.model_dump()]


rbln_actor = RBLNActor.remote()
app = Llama3_1__8B.bind(rbln_actor)
