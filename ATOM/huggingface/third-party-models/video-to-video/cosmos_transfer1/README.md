# RBLN Cosmos-Transfer1

## Prerequisites

### Submodule Update

Before proceeding, make sure to update the submodules:

```bash
git submodule update --init ./cosmos-transfer1
```

### Download Checkpoints

After updating the submodules, you need to download the checkpoints.

Follow the instructions below to download checkpoints.


1. Move to the updated submodule (Cosmos-Transfer1).

```bash
cd cosmos-transfer1
```

2. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token. Set the access token to 'Read' permission (default is 'Fine-grained').

3. Log in to Hugging Face with the access token:

```bash
huggingface-cli login
```

4. Accept the [Llama-Guard-3-8B terms](https://huggingface.co/meta-llama/Llama-Guard-3-8B).

5. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e):

> [!note]
> Before download, you should accept **NVIDIA Open Model License Agreement** for the model collection.
> Please make sure that you accept the agreement for the models below.
> - [nvidia/Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1)
> - [nvidia/Cosmos-Tokenize1-CV8x8x8-720p](https://huggingface.co/nvidia/Cosmos-Tokenize1-CV8x8x8-720p)

> [!note]
> As of `cosmos-guardrail==0.3.1`, the text guardrail uses `Qwen3Guard` and the video
> guardrail keeps only `face_blur_filter`. The SigLIP content-safety filter is disabled
> upstream, so `siglip_encoder` and `video_safety_model` are no longer compiled.

```bash
PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir ../ckpt --model 7b
```

Please verify that `--output_dir` points to the directory where you want to save the checkpoints.


Please refer to the following document for detailed instructions: [Download Checkpoints Guide](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b.md#download-checkpoints)


## Quick Test

After downloading checkpoints, you can run examples with the below scripts.

We assume that you already download the original checkpoints to `./ckpt`.

Please refer to the following contents for detailed information.


### Distilled ControlNet

```bash
python compile.py --model_name distil
python inference.py --model_name distil
```


### Single ControlNet

```bash
python compile.py --model_name single
python inference.py --model_name single
```

### Regional Prompt Example

```bash
python compile.py --model_name region
python inference.py --model_name region
```

### Multiview Example

```bash
python compile.py --model_name multiview
python inference.py --model_name multiview
```


## Compile Models

### Available Presets

Use the following command to list the available presets defined in `configs/preset.yaml`:

```bash
python compile.py
```

The available presets are:

- `4kupscaler`
- `4kupscaler_perf`
- `av`
- `av_perf`
- `distil`
- `distil_perf`
- `multi`
- `multi_perf`
- `multiview`
- `multiview_lvg`
- `multiview_lvg_perf`
- `multiview_perf`
- `region`
- `region_perf`
- `single`
- `single_perf`


You can compile models with a desired preset like below:

```bash
python compile.py --model_name <preset>
```


### NPU Resource Optimization Settings

While generating videos, a lot of RBLN devices are needed.

If you don't have enough devices, we recommend configurations that use fewer devices.

To do so, we provide NPU resource optimization settings that try to use a smaller number of devices.

The presets for NPU resource optimization are:

- `4kupscaler`
- `av`
- `distil`
- `multi`
- `multiview`
- `multiview_lvg`
- `region`
- `single`


### Latency Optimization Setting

If you have enough devices to run multiple ControlNets, we recommend the configuration that generates videos faster than the NPU resource optimization setting.

These presets have the `_perf` suffix in the name; other options are the same as in the NPU resource optimization settings.

The presets for latency optimization are:

- `4kupscaler_perf`
- `av_perf`
- `distil_perf`
- `multi_perf`
- `multiview_perf`
- `multiview_lvg_perf`
- `region_perf`
- `single_perf`


## Inference Models

Inference scripts correspond to the compile presets are also defined in `configs/preset.yaml`.

If you compiled models via:

```bash
python compile.py --model_name <preset>
```

run:

```bash
python inference.py --model_name <preset>
```

(The same preset names live alongside the inference presets in `configs/preset.yaml`.) This generates videos with the compiled models.


## Example JSON file

```json
{
    "prompt": "The video shows a sequence of images where two women are facing each other, both wearing floral headscarves and traditional-style dresses. They appear to be engaged in a conversation, with their facial expressions and body language suggesting a friendly or intimate interaction. In the background, there is a man dressed in a military uniform, standing with his back to the camera, seemingly observing the women. The setting looks like a room with posters on the wall, and the overall atmosphere is casual and domestic.",
    "input_video_path" : "assets/example1.mp4",
    "vis": {
        "control_weight": 0.2
    },
    "edge": {
        "control_weight": 0.2
    },
    "depth": {
        "input_control": "assets/example1_depth.mp4",
        "control_weight": 0.2
    },
    "seg": {
        "control_weight": 0.2
    },
    "keypoint": {
        "control_weight": 0.2
    }
}
```

The above example is `assets/multi_test.json`.

- `prompt`: The input prompt that describes the video.

- `input_video_path`: Path to a video (.mp4) file that you want to reconstruct.

- `vis`, `edge`, `depth`, `seg`, `keypoint`: ControlNets to use. If you want to use only a subset of these 5 ControlNets, remove the other(s) that you don't want to use.

- `control_weight`: The strength indicating how much a specific ControlNet affects the output video.

- `input_control`: Path to a video (.mp4) file used as the input of a ControlNet. If not provided, the ControlNet input is generated by Preprocessors.


> [!note]
> If you want to run models with other configurations or custom JSON files, please add a new preset to `configs/preset.yaml` for both compile and inference.


## Configuration Details

> [!tip]
> To use the `yq` command in the following examples, install it via pip:
> ```bash
> pip install yq
> ```

### Multi ControlNets

You can use up to 5 ControlNets at once.

`configs/preset.yaml` lists both the compile and inference presets for each mode. Inspect the desired entry with a command such as:

```bash
yq '.compile | { "multi": ."multi" }' configs/preset.yaml
````

Then follow the compile and inference steps defined in that preset entry.

### Single ControlNet

You can use a subset of the compiled ControlNets that you compiled above at inference time.

However, if you want to compile modules with only a single ControlNet, rely on the `single` (or `single_perf`) entries defined under `configs/preset.yaml`; inspect them via:

```bash
yq '.compile | { "single": ."single" }' configs/preset.yaml
```

Then follow the compile and inference steps defined in that entry.

(Please update `assets/single_test.json` to compile the desired ControlNet.)

### Regional Prompt Example

We support regional prompts, and the `region` (`region_perf`) presets defined in `configs/preset.yaml` drive both compile and inference flows for these examples. Peek at them with:

```bash
yq '.compile | { "region": ."region" }' configs/preset.yaml
```

Then follow the compile and inference steps described under that entry.

### Distilled ControlNet

With the distilled ControlNet, we can generate videos in a single diffusion step; therefore, the inference time is reduced dramatically.

To compile a distilled ControlNet and other necessary modules, follow the `distil` / `distil_perf` entries in `configs/preset.yaml`, which also include the inference metadata you should reuse. Inspect the parameters via:

```bash
yq '.compile | { "distil": ."distil" }' configs/preset.yaml
```

Then execute the compile and inference steps that that YAML entry describes.

> [!note]
> Currently, NVIDIA provides the Edge ControlNet only for the distilled version.

### 4K Upscaler

With the 4K upscaler, you can upscale 720p-resolution videos to 4K-resolution videos.

To compile modules for the 4K upscaler, use the `4kupscaler` (`4kupscaler_perf`) preset defined in `configs/preset.yaml`, which also records the matching inference definition. Preview the settings with:

```bash
yq '.compile | { "4kupscaler": ."4kupscaler" }' configs/preset.yaml
```

Then perform the compile and inference commands that the YAML entry lists.

### Sample-AV

There are two more modalities specialized for autonomous vehicle applications.

To compile modules for Sample-AV, pick the `av` (`av_perf`) preset inside `configs/preset.yaml`, then run:

```bash
yq '.compile | { "av": ."av" }' configs/preset.yaml
```

Then follow the compile and inference commands defined in that entry.

### Sample-AV Single to Multiview

With this model, you can generate multi-view video from a single video.

To compile modules for Sample-AV Single to Multiview, use the `multiview` (`multiview_perf`) preset from `configs/preset.yaml`, which also lists the inference configuration. Inspect it via:

```bash
yq '.compile | { "multiview": ."multiview" }' configs/preset.yaml
```

Then follow the compile and inference commands encapsulated in that preset.

### Sample-AV Single to Multiview Video2World

With a multi-view video generated with the Single to Multiview model, you can further extend the video with this model.

To compile modules for Sample-AV Single to Multiview Video2World, use the `multiview_lvg` (`multiview_lvg_perf`) preset from `configs/preset.yaml`, which again bundles the inference settings. Preview the entry with:

```bash
yq '.compile | { "multiview_lvg": ."multiview_lvg" }' configs/preset.yaml
```

Then carry out the compile and inference flows that that entry specifies.
