# C++ Model Zoo

Standalone C++ example suites.

## Default policy (run with minimal flags)

The modelzoo aims to support a simple, reproducible flow where **defaults are sufficient** once you follow the basic build/compile steps.

Common conventions:

- `compile.py` produces artifacts in the **current working directory**.
- C++ binaries accept `--model <path>`.
  - In the modelzoo, `--model` defaults to `.` so you can run binaries without flags after compile.
- Some suites use sample inputs bundled in the directory.
- Some suites provide `pre_process.py` / `post_process.py`.
  - `pre_process.py`: prepares inputs before running the binary.
  - `post_process.py`: validates/decodes outputs after running the binary.

## Quickstart (per suite)

### image_classification

- Compile (default model):

```sh
python3 compile.py
```

Creates `resnet18.rbln`.

- Build:

```sh
mkdir -p build && cd build && cmake .. && make -j"$(nproc)"
```

- Run (no flags):

```sh
./build/image_classification
```

Uses `tabby.jpg` and finds a `.rbln` under `.`.

### object_detection

- Compile (default model):

```sh
python3 compile.py
```

Creates `yolov8n.rbln`.

- Build:

```sh
mkdir -p build && cd build && cmake .. && make -j"$(nproc)"
```

- Run (no flags):

```sh
./build/object_detection
```

Uses `people4.jpg` and finds a `.rbln` under `.`.

### text_generation

- Prepare inputs:

```sh
python3 pre_process.py
```

Creates `c_input_ids.bin`.

- Compile (default model-id):

```sh
python3 compile.py
```

Creates `Meta-Llama-3-8B-Instruct/` containing `prefill.rbln` and `decoder_batch_1.rbln`.

- Build:

```sh
mkdir -p build && cd build && cmake .. && make -j"$(nproc)"
```

- Run (no flags):

```sh
./build/text_generation
```

- Post-process:

```sh
python3 post_process.py
```

Decodes `c_text2text_generation_gen_id.bin`.
