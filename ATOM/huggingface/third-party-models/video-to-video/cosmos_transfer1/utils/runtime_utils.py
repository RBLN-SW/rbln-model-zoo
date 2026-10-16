from typing import Dict, Optional

import rebel
import torch
from cosmos_transfer1.utils import log
from optimum.rbln.utils.runtime_utils import RBLNPytorchRuntime


class RBLNRuntimeVAE(RBLNPytorchRuntime):
    """
    An implementation of Cosmos Transfer1's VAE for RBLN Runtime.

    Args:
        compiled_model: The model compiled with RBLN.
        rbln_config: The configuration for creating the RBLN runtime.
    """

    def __init__(
        self,
        compiled_model: rebel.RBLNCompiledModel,
        rbln_config: Optional[Dict] = None,
    ):
        self.compiled_model = compiled_model
        self.rbln_config = rbln_config
        self._runtime = None

        self.dtype = torch.float32

    @property
    def runtime(self):
        """
        Returns:
            The RBLN runtime for the model. If the runtime is not created, it will
            be created automatically.
        """
        if self._runtime is None:
            raise ValueError(
                "Runtime is not created. Please set `create_runtimes=True` first."
            )
        return self._runtime

    def create_runtime(self):
        """
        Creates the RBLN runtime for the model.
        """
        if self._runtime is None:
            if self.rbln_config is None:
                device = None
            else:
                device = self.rbln_config.get("device", None)
            self._runtime = self.compiled_model.create_runtime(
                tensor_type="pt", device=device
            )
        else:
            log.info("Runtime is created already.")
