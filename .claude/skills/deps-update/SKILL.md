---
name: deps-update
description: Add or change a dependency in RBLN Model Zoo the right way — edit the example requirements.txt and let the consolidation flow refresh pyproject.toml / uv.lock. Triggers on "add a package", "bump optimum-rbln/vllm-rbln", or uv.lock / unified-requirements questions.
---

# Dependency update

## When to use

- Adding or bumping a package that an example needs.
- Changing the `optimum-rbln` / `vllm-rbln` pins, `uv.lock`, or `unified_requirements.txt`.

## Steps

1. Edit the example's `requirements.txt` — the per-example source of truth. Add the
   package unpinned, keeping the `--extra-index-url https://download.pytorch.org/whl/cpu`
   header.
2. Regenerate the consolidated set: `python utils/generate_unified_requirements.py --uv-lock`.
   This folds every example's `requirements.txt` into the root `pyproject.toml` (the `models`
   extra) and refreshes `uv.lock`.
3. Commit `pyproject.toml` and `uv.lock` together. Never hand-edit `uv.lock`.

## Notes

- `unified_requirements.txt` is a generated artifact and is not committed.
- `uv.lock` is the resolved source of truth for the RBLN pins. A new resolved
  `optimum-rbln` / `vllm-rbln` updates it even when the pin spec is unchanged.
- Keep platform-specific packages behind their existing environment markers.
