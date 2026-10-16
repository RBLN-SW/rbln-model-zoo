# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import attrs
import rebel
import torch
from optimum.rbln import RBLNBaseModel
from optimum.rbln.configuration_utils import RBLNModelConfig
from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import (
    RBLNDecoderOnlyModelForCausalLM,
)
from optimum.rbln.utils.logging import get_logger
from transformers import GenerationConfig, MistralConfig, PretrainedConfig

from .configuration_mistral_upsampler import RBLNMistralNeMoForTextUpsamplerConfig
from .mistral_upsampler_architecture import MistralNeMoForTextUpsamplerWrapper
from .model.modeling_mistral_upsampler import MistralNeMoForTextUpsampler

logger = get_logger()


class RBLNMistralNeMoForTextUpsampler(RBLNDecoderOnlyModelForCausalLM):
    _rbln_config_class = RBLNMistralNeMoForTextUpsamplerConfig
    _decoder_wrapper_cls = MistralNeMoForTextUpsamplerWrapper

    def __init__(
        self,
        models: List[rebel.Runtime],
        config: Optional[PretrainedConfig],
        rbln_config: RBLNModelConfig,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        subfolder: str = "",
        rbln_compiled_models: Optional[rebel.RBLNCompiledModel] = None,
        rbln_submodules: List["RBLNBaseModel"] = [],
        **kwargs,
    ):
        self.model = models
        self.config = config
        self.rbln_config = rbln_config
        if not rbln_config.is_frozen():
            raise RuntimeError(
                "`rbln_config` must be frozen. Please call `rbln_config.freeze()` first."
            )

        self.compiled_models = rbln_compiled_models

        if self.can_generate():
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.ckpt_dir, use_fast=True)
            gen_config = {
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": 10,
            }
            self.generation_config = GenerationConfig.from_dict(gen_config)
        else:
            self.generation_config = None

        # self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        if self.generation_config is not None:
            self.generation_config.use_cache = True

        self.device = torch.device("cpu")
        self.training = False
        self.dtype = rbln_config.torch_dtype

        # FIXME :: model_save_dir is not used after initialized. (This can be used when save/load)
        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting it
        # would end-up removing the directory containing the underlying RBLN model.
        self._model_save_dir_tempdirectory_instance = None
        if isinstance(model_save_dir, TemporaryDirectory):
            self._model_save_dir_tempdirectory_instance = model_save_dir
            self.model_save_dir = Path(model_save_dir.name)
        elif isinstance(model_save_dir, str):
            self.model_save_dir = Path(model_save_dir)
        else:
            self.model_save_dir = model_save_dir
        self.subfolder = subfolder

        self.rbln_submodules = rbln_submodules
        self.__post_init__(**kwargs)

    @classmethod
    def get_pytorch_model(
        cls,
        model_id: str,
        # Some rbln-config should be applied before loading torch module (i.e. quantized llm)
        rbln_config: Optional[RBLNMistralNeMoForTextUpsamplerConfig] = None,
        **kwargs,
    ) -> MistralNeMoForTextUpsampler:
        max_seq_len = rbln_config.max_seq_len
        batch_size = rbln_config.batch_size

        model = MistralNeMoForTextUpsampler(
            model_id=model_id,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            **kwargs,
        )

        original_config = attrs.asdict(model.config)
        config = MistralConfig.from_dict(original_config)

        # Synchronization with original RBLNDecoderOnlyForCausalLMConfig
        sync_kwargs = {
            "n_head": getattr(config, "n_heads"),
            "num_key_value_heads": getattr(config, "n_kv_heads"),
            "num_hidden_layers": getattr(config, "n_layers"),
            "hidden_size": getattr(config, "dim"),
        }
        config.update(sync_kwargs)
        config.sliding_window = None
        model.config = config

        for module in model.model.modules():
            if getattr(module, "config", None) is not None:
                module.config = config

        model.lm_head = model.model.output
        del model.model.output
        return model
