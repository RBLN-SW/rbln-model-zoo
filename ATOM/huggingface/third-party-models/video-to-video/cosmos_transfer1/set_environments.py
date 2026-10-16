import importlib.util
import os
import sys
from unittest.mock import MagicMock


def set_environments():
    custom_dir = os.path.abspath("rbln_module")
    target_import_prefix = "cosmos_transfer1.diffusion.module"

    for filename in os.listdir(custom_dir):
        if filename.endswith(".py"):
            module_name = filename[:-3]
            full_target_name = f"{target_import_prefix}.{module_name}"
            custom_file_path = os.path.join(custom_dir, filename)
            spec = importlib.util.spec_from_file_location(
                full_target_name, custom_file_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_target_name] = module
            spec.loader.exec_module(module)

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosmos-transfer1")
    )

    mock_module = MagicMock()
    mock_module.is_initialized.return_value = False
    mock_module.get_context_parallel_world_size.return_value = 1
    sys.modules["amp_C"] = mock_module
    sys.modules["apex"] = mock_module
    sys.modules["apex"].__spec__ = mock_module
    sys.modules["apex.multi_tensor_apply"] = mock_module
    sys.modules["megatron.core"] = mock_module
    sys.modules["megatron.core"].parallel_state = mock_module
    sys.modules["transformer_engine"] = mock_module
    sys.modules["transformer_engine.pytorch"] = mock_module
    sys.modules["transformer_engine.pytorch.attention"] = mock_module
    sys.modules["transformer_engine.pytorch.attention.rope"] = mock_module
    sys.modules["transformer_engine.pytorch.attention.dot_product_attention"] = (
        mock_module
    )
    sys.modules[
        "transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention"
    ] = mock_module

    inference_utils_import_path = "cosmos_transfer1.diffusion.inference.inference_utils"
    custom_inference_utils_path = os.path.abspath("rbln_inference/inference_utils.py")
    spec = importlib.util.spec_from_file_location(
        inference_utils_import_path, custom_inference_utils_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[inference_utils_import_path] = module
    spec.loader.exec_module(module)


set_environments()
