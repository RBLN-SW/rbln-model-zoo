<div align="center">

<img src="rbln-model-zoo-banner.png" alt="RBLN Model Zoo" width="600">

*500+ models · Compile once, run anywhere · AI model serving on RBLN NPUs*

# RBLN Model Zoo

[![models](https://img.shields.io/badge/models-Model%20Zoo-10B981?style=flat-square)](https://rebellions.ai/developers/model-zoo)
[![docs](https://img.shields.io/badge/docs-latest-8B5CF6?style=flat-square)](https://docs.rbln.ai)

[![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-F59E0B?style=flat-square)](https://docs.rbln.ai/supports/version_matrix.html)
[![ubuntu](https://img.shields.io/badge/Ubuntu-22.04%20%7C%2024.04-E95420?style=flat-square&logo=ubuntu)](https://docs.rbln.ai/supports/version_matrix.html)
[![rhel](https://img.shields.io/badge/RHEL-9.4%20%7C%209.6-EE0000?style=flat-square&logo=redhat)](https://docs.rbln.ai/supports/version_matrix.html)
[![support](https://img.shields.io/badge/NPU-Support%20Matrix-10B981?style=flat-square)](https://docs.rbln.ai/supports/version_matrix.html)

</div>

---

## Quick Start

```bash
# Install compiler
pip install -i https://pypi.rbln.ai/simple rebel-compiler==0.10.2

# Navigate to model directory
cd huggingface/transformers/text2text-generation/llama/llama3.1-8b

# Install dependencies
pip install -r requirements.txt

# Compile and run
python compile.py && python inference.py
```

> [!NOTE]
> The versions pinned above match what this repo was tested with as of **2026-03-27**; a newer stable release may already be available. For current versions and install steps, see the [RBLN installation guide](https://docs.rbln.ai/latest/getting_started/installation_guide.html).

> [!TIP]
> For models that support configuration presets, use `--model_name <preset>` to specify model-specific configurations. See each model's README for available presets.

> [!IMPORTANT]
> A [RBLN portal account](https://docs.rbln.ai/getting_started/installation_guide.html) is required to install `rebel-compiler` from PyPI.

---

## Ecosystems & APIs

Select the ecosystem or API for your AI serving workload on RBLN NPUs.

### Python

| Ecosystem | # Models | Key packages |
|-----------|----------|---------------|
| Hugging Face | 150+ | transformers, diffusers |
| PyTorch | 250+ | torch |
| TensorFlow | 75+ | keras, tensorflow |

### Other

**C API** — C/C++ inference bindings. Install via [APT](https://docs.rbln.ai/software/api/language_binding/c/installation.html), then build from source.

---

## Deployment

### vLLM-RBLN

Compile a model from the Model Zoo, then deploy with:

```bash
# Compile
python compile.py

# Install vLLM-RBLN
pip3 install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://wheels.vllm.ai/0.13.0/cpu \
  vllm-rbln==0.10.2
```

> [!NOTE]
> The versions pinned above match what this repo was tested with as of **2026-03-27**; a newer stable release may already be available. For current versions and install steps, see the [RBLN installation guide](https://docs.rbln.ai/latest/getting_started/installation_guide.html).

```python
# Import
from vllm import LLM, SamplingParams

# Load model and generate
llm = LLM(model="Llama-3.1-8B-Instruct")
out = llm.generate(["Hello"], SamplingParams(max_tokens=64))
print(out[0].outputs[0].text)
```

- [vLLM-RBLN](https://docs.rbln.ai/software/model_serving/vllm_support/vllm-rbln.html) — LLM serving on RBLN NPUs
- [Triton](https://docs.rbln.ai/software/model_serving/nvidia_triton_inference_server/installation.html) — Triton Inference Server
- [TorchServe](https://docs.rbln.ai/software/model_serving/torchserve/torchserve.html) — PyTorch model serving

---

## Links

- [CHANGELOG](CHANGELOG.md) — Release history
- [Issues](https://github.com/RBLN-SW/rbln-model-zoo/issues) — Report issues, request features, or request new model support.
