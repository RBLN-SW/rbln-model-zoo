# Text Generation (C/C++ binding APIs)

## Default run scenario

- `python3 pre_process.py` → `c_input_ids.bin`
- `python3 compile.py` → `Meta-Llama-3-8B-Instruct/` (contains `prefill.rbln`, `decoder_batch_1.rbln`)
- build with CMake
- `./build/text_generation`
- `python3 post_process.py` → decodes `c_text2text_generation_gen_id.bin`
