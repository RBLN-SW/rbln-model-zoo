# Third-Party Models

This directory hosts models that are published on the **Hugging Face Hub** but are **not implemented with Hugging Face’s official frameworks** (`diffusers` or `transformers`).
Instead, these entries rely on **external or vendor-specific codebases** (e.g., personal GitHub repositories, research implementations).

---

## Purpose

* Provide a **consistent integration layer** for models outside of Hugging Face’s native frameworks.
* Ensure Hugging Face Hub weights remain **usable and reproducible** in the model zoo.
* Maintain a **clear separation** between official Hugging Face frameworks and third-party integrations.

---

## Conventions

* **Naming**
  Use lowercase, hyphen-separated names (`my-model`), consistent with zoo-wide standards.

* **Layout**
  Mirror the task-based hierarchy (`category/model`) used under `diffusers/` and `transformers/`.

* **Adapters**
  Each model must provide thin adapter scripts (`compile.py`, `inference.py`, `requirements.txt`) to align with zoo execution standards.

---

## Lifecycle & Deprecation

* Entries here are treated as **temporary bridges**.
* If a model later receives **native `diffusers` or `transformers` support**, the third-party entry should follow this deprecation workflow:

  ### Deprecation Workflow
  1. **Mark as Deprecated**
     - Add a clear deprecation notice at the top of the model's README (e.g., "⚠️ This model is deprecated. Please use the official implementation in `diffusers`/`transformers` [here](link-to-official-model).").
     - Add a warning banner to all adapter scripts (`compile.py`, `inference.py`) indicating deprecation.
  2. **Announce Deprecation**
     - Open a PR with the deprecation notice and warning banners.
     - Tag relevant maintainers and contributors.
  3. **Grace Period**
     - Allow a standard grace period of **30 days** after deprecation notice before removal.
  4. **Migration**
     - Migrate the model to the official directory (`diffusers/` or `transformers/`).
     - Update documentation to point users to the new location.
  5. **Migration PR Checklist**
     - [ ] Add deprecation notice to README.
     - [ ] Add warning banners to adapter scripts.
     - [ ] Link to official implementation.
     - [ ] Announce deprecation in PR.
     - [ ] Wait for grace period to elapse.
     - [ ] Remove third-party entry after migration.

---

## Scope of Inclusion

This directory is reserved strictly for:

* Models with Hugging Face weights but **no official `diffusers`/`transformers` support**.
* Third-party vendor or research frameworks that require minimal adapters for integration.

Not included:

* Purely local models without Hugging Face Hub hosting.
* Models already integrated upstream in `diffusers` or `transformers`.

---

✅ **Policy Check**:
When adding a model here, ask:

* *Does it live on Hugging Face Hub?*
* *Is it unsupported in official Hugging Face frameworks?*
* *Does it require a thin adapter for reproducibility?*

If all answers are **yes**, it belongs here.
