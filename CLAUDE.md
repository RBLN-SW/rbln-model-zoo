# Claude Guide — RBLN Model Zoo

## Guiding principle

RBLN Model Zoo is a collection of runnable examples that compile and run public models on the
RBLN NPU — ATOM™+ (`RBLN-CA22`) and ATOM™-Max (`RBLN-CA25`). One rule governs the whole
repository:

**Keep each example faithful to its upstream, public usage. Confine every divergence to the
RBLN compile-and-run step, and preserve the model's semantics — the same model ID, the same
inputs, and the same pre- and post-processing.** An example is a thin adapter: it loads a
public model, compiles it for the RBLN NPU, and runs it. It must not reimplement modeling (a
custom `forward`, or tensor shapes a native user would not write). When an example appears to
require this, the work belongs upstream and should be reported there.

"Faithful" is bounded, not literal. A few divergences are inherent and expected:

- **The RBLN integration itself** — `AutoModelFor*` becomes `RBLNAutoModelFor*`, with an
  `export=True` compile step and `save_pretrained()` (or a saved `.rbln` artifact).
- **Compilation constraints** — static shapes, a fixed batch size, and `max_length` padding,
  because the compiled graph is not dynamic. Keep these minimal and explicit.

Where the modeling lives depends on the path:

- **Hugging Face (`transformers`, `diffusers`)** — modeling comes from Optimum RBLN
  (`RBLNAutoModelFor*`). The example only consumes it, so the repository stays thin.
- **PyTorch, PyTorch Dynamo, TensorFlow, C API, vLLM** — there is no Optimum RBLN layer; the
  example uses the RBLN Compiler directly (`rebel.compile_from_torch`, or
  `torch.compile(backend="rbln")`) and keeps the model in its native framework form.
- **Hugging Face models outside `transformers` / `diffusers`** — published on the Hub but
  implemented in their own repository or package. These live under
  `huggingface/third-party-models/` and necessarily carry vendored modeling. See
  "Third-party models" below.

An example that mirrors public usage stays clear, portable, and easy to trust: a reader can
compare it against the model card and see only the RBLN integration step.

## Ecosystem and repository roles

The RBLN SDK is layered, and each repository has a distinct charter. Knowing which component
owns a concern is the key to triage, because an issue usually surfaces in one repository but
is owned by another.

- **RBLN Compiler** (`rebel-compiler`, imported as `rebel`) — compiles a Torch graph into an
  RBLN NPU binary and owns the low-level compile and runtime path. A compilation assertion or
  code-generation error originates here.
- **Optimum RBLN** (`optimum-rbln`) — the Hugging Face integration library and the canonical
  owner of the model API. `RBLNAutoModelFor*` classes export and run `transformers` and
  `diffusers` models; this is where model architecture, configuration, and the compile path
  are adapted to the NPU, and where the `transformers` version is pinned.
- **RBLN Model Zoo** (this repository) — the public-facing examples. Thin adapters that
  consume Optimum RBLN (Hugging Face path) or the RBLN Compiler directly (other frameworks),
  faithful to native usage. It demonstrates how to run models on the RBLN NPU; it does not
  implement them.
- **rbln-exec** (`rbln_executor`) — the functional and numerical-correctness check. It tracks
  Optimum RBLN's modeling and compares native (eager) output against RBLN-compiled output to
  catch regressions. Because both consume the same Optimum RBLN, it stays consistent with the
  Model Zoo examples.

Dependency and responsibility flow downstream:

```
RBLN Compiler  ←  Optimum RBLN (canonical model API)  ←  RBLN Model Zoo (native-faithful examples)
                                                      ←  rbln-exec       (tracks Optimum RBLN; verifies output)
```

Optimum RBLN defines the model API; the Model Zoo and rbln-exec follow it. When Optimum RBLN
raises its `transformers` pin, examples written for the previous API can fail — so a failure
can surface in the Model Zoo while its fix belongs upstream. **Ownership follows the failing
frame:**

- An example using an outdated native API (a renamed preprocessor, a removed pipeline task, a
  changed public call) — fix it **here**, by matching current native usage.
- A model wrapper, configuration, or compile path under `optimum/rbln/...` — **Optimum RBLN**.
- A compilation assertion during compile — **RBLN Compiler**.
- A numerical mismatch between native and compiled output — **Optimum RBLN** or **RBLN
  Compiler**; do not mask it in the example.
- When fixing the example would mean diverging from its public, native form, treat it as an
  upstream issue to report.

## Applying the principle to an example

- **Structure** — examples are organized as `framework/task/model`. Hugging Face and PyTorch
  examples split into `compile.py` (export, then `save_pretrained()` or a saved `.rbln`) and
  `inference.py` (load, run, print or save), so compilation runs once. PyTorch Dynamo
  examples use a single `main.py` with `torch.compile(..., backend="rbln")` (keep
  `import rebel` to register the backend). Match the nearest sibling rather than introducing a
  new layout.
- **Consume, don't reimplement** — Hugging Face examples use `optimum.rbln`
  `RBLNAutoModelFor*`; PyTorch examples use `rebel.compile_from_torch`. The remaining code
  (loading the processor, building inputs, decoding outputs) should read like native usage.
- **Mirror the upstream docs example** — when writing or updating an example, follow the
  official usage in the Hugging Face docs site for the package it builds on:
  `transformers` (`https://huggingface.co/docs/transformers/model_doc/<model>`), `diffusers`
  (`https://huggingface.co/docs/diffusers/api/pipelines/<name>`), `tokenizers`, and so on.
  Use it as the reference for the model/pipeline class, tokenizer/processor, input
  construction, and output decoding. These pages track the current library API, so they
  reflect the version we pin — a more reliable reference than a Hub model card, which can lag
  the API (e.g. still showing a removed pipeline task). Keep only the RBLN compile-and-run step
  as the divergence. This is also the first reference when an example breaks after a version bump:
  reconcile the call against the model's current doc example before reshaping it from the
  traceback, which on its own can lead to a fix that diverges from idiomatic usage.
- **Preserve the contract** — keep real model IDs, real example inputs, and native pre- and
  post-processing; these let the example be compared against upstream.
- **Print results plainly** — print each result on its own line with an inline, capitalized
  label, using an f-string (`print(f"Result: {value}")`). Always label the value; never print
  it bare. Render tensors as plain Python values (`.item()`, `.tolist()`) rather than
  `tensor(...)`.
- **Dependencies** — each example's `requirements.txt` lists only that example's dependencies,
  unpinned, with the `--extra-index-url https://download.pytorch.org/whl/cpu` header. The root
  `pyproject.toml` and `uv.lock` are the consolidated, resolved source of truth and hold the
  RBLN pins. See the `deps-update` skill; never edit `uv.lock` by hand.

## Third-party models

`huggingface/third-party-models/` is home to models published on the Hugging Face Hub but
**not implemented in the `transformers` or `diffusers` libraries** — they ship in their own
repository or package. With no library to supply the modeling, these examples carry vendored
RBLN integration code and hold more than a thin adapter. That is inherent to this path, not a
defect.

The category is permanent. "Temporary" applies only to an **individual entry**: if a model's
modeling is later integrated into `transformers` or `diffusers` upstream, the entry graduates
to the native path (where Optimum RBLN can frameworkize it). Many models are never integrated
and stay here indefinitely, which is correct. When an entry graduates, follow the deprecation
workflow in the directory's `README.md`.

## Working in this repository

- Verify an example before reporting it as working: `python compile.py && python inference.py`
  (or `python main.py`). Compilation runs on the host; the run step requires an RBLN NPU.
- The lint gate covers import sorting and formatting only — see the `ruff-lint` skill.
- Do not commit build artifacts (`.rbln` files, `.rbln_zoo_cache/`, or the generated
  `unified_requirements.txt`).
- Canonical repository URL: `https://github.com/RBLN-SW/rbln-model-zoo`. The brand prefix is
  "RBLN"; the packages are `optimum-rbln` and `vllm-rbln`; the device cards are ATOM™+
  (`RBLN-CA22`) and ATOM™-Max (`RBLN-CA25`).

## Extending this guide

This file is the always-on guide: the principle, the repository roles, and triage. When a
concrete procedure recurs — scaffolding an example, updating a dependency, passing the lint
gate — document it as a skill under `.claude/skills/<name>/SKILL.md`, which loads only when
its description matches the task. Keep this file focused on direction, and keep procedures in
skills.

A minimal `SKILL.md`:

```markdown
---
name: <skill-name>
description: <one-line trigger description; keywords decide auto-activation>
---

# <Skill name>

## When to use
- <recurring task pattern this skill handles>

## Steps
1. <first step>

## Notes
- <gotchas, related links>
```
