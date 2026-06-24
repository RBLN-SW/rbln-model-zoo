---
name: ruff-lint
description: Make a change pass the RBLN Model Zoo lint CI gate — ruff import-sort (--select I) and ruff format. Triggers before committing/opening a PR, or on a failing "Lint" / ruff CI check.
---

# Ruff lint gate

## When to use

- Before committing or opening a PR that touches `.py` files.
- When the `Lint` / ruff CI check fails.

## Steps

1. Sort imports: `ruff check . --select I --fix`.
2. Format: `ruff format .`.
3. Confirm a clean gate the way CI checks it (read-only): `ruff check . --select I --diff`
   and `ruff format . --check` — both should report no changes.

## Notes

- Only import sorting (`I`) and formatting are enforced — not Ruff's default `F` / `E`
  rules. Don't "fix" lint the gate doesn't check.
- Editing a string or comment can re-wrap a line, so re-run `ruff format .` afterward.
