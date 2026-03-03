<div align="center">

<img src="rbln-model-zoo-banner.png" alt="RBLN Model Zoo" width="600">

*AI serving with RBLN NPUs · Compile once, Run anywhere · 500+ models — start here*

# RBLN Model Zoo

[![models](https://img.shields.io/badge/models-Model%20Zoo-10B981?style=flat-square)](https://rebellions.ai/developers/model-zoo)
[![docs](https://img.shields.io/badge/docs-latest-8B5CF6?style=flat-square)](https://docs.rbln.ai)

</div>

---

## Quick Start

```bash
# 1. Install RBLN compiler
pip install -i https://pypi.rbln.ai/simple rebel-compiler

# 2. Pick a model & install deps
cd huggingface/transformers/text2text-generation/qwen/qwen2.5-7b
pip install -r requirements.txt

# 3. Compile → Run
python compile.py && python inference.py
```

Compile once, run anywhere.

> [RBLN portal account](https://docs.rbln.ai/getting_started/installation_guide.html) required

---

## Frameworks

Hugging Face, PyTorch, TensorFlow — your starting point for AI serving on RBLN NPUs.

| Framework | Models |
|-----------|--------|
| Hugging Face | 150+ |
| PyTorch | 250+ |
| TensorFlow | 75+ |

**Example** — `huggingface/transformers/text2text-generation/qwen/qwen2.5-7b`

**C API** — C/C++ inference bindings. Install via [APT](https://docs.rbln.ai/software/api/language_binding/c/installation.html), then compile and run from `cpp/`.

---

## Platform

[![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-F59E0B?style=flat-square)](https://docs.rbln.ai/supports/version_matrix.html)
[![ubuntu](https://img.shields.io/badge/Ubuntu-22.04%20%7C%2024.04-E95420?style=flat-square&logo=ubuntu)](https://docs.rbln.ai/supports/version_matrix.html)
[![rhel](https://img.shields.io/badge/RHEL-9.4%20%7C%209.6-EE0000?style=flat-square&logo=redhat)](https://docs.rbln.ai/supports/version_matrix.html)
[![support](https://img.shields.io/badge/NPU-Support%20Matrix-10B981?style=flat-square)](https://docs.rbln.ai/supports/version_matrix.html)

---

## Deployment

Compile once, run anywhere.

**vLLM-RBLN** — compile from Model Zoo, then serve:

```bash
# 1. Compile (from model zoo)
cd huggingface/transformers/text2text-generation/qwen/qwen2.5-7b
python compile.py  # → Qwen2.5-7B-Instruct/

# 2. Install & serve
pip install vllm-rbln
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen2.5-7B-Instruct')
out = llm.generate(['Hello'], SamplingParams(max_tokens=64))
print(out[0].outputs[0].text)
"
```

[vLLM-RBLN](https://docs.rbln.ai/software/model_serving/vllm_support/vllm-rbln.html) · [Triton](https://docs.rbln.ai/software/model_serving/nvidia_triton_inference_server/installation.html) · [TorchServe](https://docs.rbln.ai/software/model_serving/torchserve/torchserve.html)

---

**Resources**

- [CHANGELOG](CHANGELOG.md)
- [Issues](https://github.com/RBLN-SW/rbln-model-zoo/issues) — have questions? Open an issue.
