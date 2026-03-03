# RBLN Model Zoo

> ML models for [ATOM NPU](https://rebellions.ai/rebellions-product/atom-2). Each model: `compile.py` → `inference.py`.

[![docs](https://img.shields.io/badge/docs-latest-blue?style=flat-square)](https://docs.rbln.ai)
[![models](https://img.shields.io/badge/models-160+-green?style=flat-square)](https://rebellions.ai/developers/model-zoo)
[![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-orange?style=flat-square)](https://docs.rbln.ai/supports/version_matrix.html)

---

## Quick Start

```bash
pip install -i https://pypi.rbln.ai/simple rebel-compiler
cd huggingface/transformers/text2text-generation/qwen/qwen2.5-7b
pip install -r requirements.txt
python compile.py && python inference.py
```

> Requires [RBLN portal account](https://docs.rbln.ai/getting_started/installation_guide.html).

---

## Frameworks

| Framework | Models | Install |
|-----------|--------|---------|
| Hugging Face | 120+ | `pip install -r <model_dir>/requirements.txt` |
| PyTorch | 23 | `pip install -r pytorch/<dir>/requirements.txt` |
| TensorFlow | 5 | `pip install -r tensorflow/<dir>/requirements.txt` |
| C/C++ | 3 | [APT](https://docs.rbln.ai/software/api/language_binding/c/installation.html) |

---

## Requirements

| | Supported |
|---|-----------|
| Python | 3.10, 3.11, 3.12, 3.13 |
| OS | Ubuntu 22.04/24.04, RHEL 9.4/9.6 |
| NPU | RBLN-CA12, CA22, CA25 |

[Full Support Matrix →](https://docs.rbln.ai/supports/version_matrix.html)

---

## Model Categories

**Hugging Face** — Text gen (50) · VLM (14) · Reranker (13) · T2I (11) · I2I (8) · Video (6) · Speech (3) · Vision (4)

**PyTorch** — YOLO, UNet, BERT, BGE, ConvTasNet

**Serving** — Triton · TorchServe · RayServe (Llama3, YOLOv8, ResNet)

---

## Deployment

[vLLM](https://docs.rbln.ai/software/model_serving/vllm_support/vllm-rbln.html) · [Triton](https://docs.rbln.ai/software/model_serving/nvidia_triton_inference_server/installation.html) · [TorchServe](https://docs.rbln.ai/software/model_serving/torchserve/torchserve.html)

---

## Links

- [Documentation](https://docs.rbln.ai)
- [Model Catalog](https://rebellions.ai/developers/model-zoo)
- [Tutorials](https://docs.rbln.ai/software/optimum/tutorial/llama3-8B.html)
- [CHANGELOG](CHANGELOG.md) · [Issues](https://github.com/RBLN-SW/rbln-model-zoo/issues)
