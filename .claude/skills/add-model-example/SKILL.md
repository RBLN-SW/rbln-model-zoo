---
name: add-model-example
description: Scaffold a new RBLN Model Zoo example with the correct directory placement and adapter scripts. Triggers on "add a model", "new example", "port model X to the zoo", or requests to create compile/inference scripts for a model.
---

# Add a model example

## Principle

An example is a thin adapter: it loads a public model, compiles it for the RBLN NPU, and
runs it. The only RBLN-specific seam is the compile-and-run call (`optimum.rbln`
`RBLNAutoModelFor*`, or `rebel.compile_from_torch`); the rest should read like native Hugging
Face or framework code. Do not reimplement modeling here — that logic belongs in Optimum RBLN.
When an example appears to need custom modeling, treat it as an upstream gap to report, not
modeling to write by hand.

## When to use

- Adding a new model to RBLN Model Zoo, or porting one from another framework.
- Creating the `compile.py` / `inference.py` / `main.py` and `requirements.txt` adapter set.

## Steps

1. Place the example at `framework/task/model`, using a lowercase, hyphen-separated leaf name
   that mirrors an existing sibling's path shape.
2. Pick the adapter shape by framework:
   - **Hugging Face** → `optimum.rbln` `RBLNAutoModelFor*`: `compile.py`
     (`from_pretrained(export=True, rbln_*=...)` + `save_pretrained()`) and `inference.py`
     (`from_pretrained(export=False)`, run, print or save).
   - **PyTorch** → `compile.py` (`rebel.compile_from_torch(model, input_info)` → save `.rbln`)
     and `inference.py` (`rebel.Runtime(...)`).
   - **PyTorch Dynamo** → a single `main.py` with
     `torch.compile(model, backend="rbln", dynamic=False, options={...})`; keep `import rebel`
     to register the backend.
3. Open the Hugging Face docs page for the package it builds on — `transformers`
   (`https://huggingface.co/docs/transformers/model_doc/<model>`), `diffusers`
   (`https://huggingface.co/docs/diffusers/api/pipelines/<name>`), `tokenizers`, etc. — and
   mirror its usage example: model/pipeline class, tokenizer/processor, input construction,
   and output decoding. Prefer it over a Hub model card, which can lag the API.
4. Copy the closest sibling example and adapt it — keep the `argparse` argument names and the
   `choices` lists for variants.
5. Write `requirements.txt`: only this example's dependencies, unpinned, with the
   `--extra-index-url https://download.pytorch.org/whl/cpu` header.
6. Pass the lint gate, then verify: `ruff check . --select I --fix && ruff format .`, then
   `python compile.py && python inference.py` (or `python main.py`).

## Notes

- Adapters only — no framework or training code in the repository.
- Print results plainly: one value per line with an inline, capitalized label
  (`print(f"Result: {value}")`). Always label the value (never print it bare), and render
  tensors as plain Python values (`.item()`, `.tolist()`), not `tensor(...)`.
- Don't add the dependency to the root `pyproject.toml` / `uv.lock` by hand; the consolidation
  flow folds `requirements.txt` in (see `deps-update`).
- Don't commit build outputs (`.rbln`, `.rbln_zoo_cache/`).
- Hugging Face models that are not in the `transformers` / `diffusers` libraries (they ship
  in their own repo or package) go under `huggingface/third-party-models/`, which carries
  vendored modeling by necessity. This is a permanent category; an entry migrates to the
  native path only if its modeling is later integrated into `transformers` / `diffusers`.
