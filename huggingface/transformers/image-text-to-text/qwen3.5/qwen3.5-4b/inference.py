import io
import os

import requests
from optimum.rbln import RBLNAutoModelForImageTextToText
from PIL import Image
from transformers import AutoProcessor

IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"


def load_image(url):
    resp = requests.get(url, headers={"Accept": "image/jpeg"}, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def main():
    model_id = "Qwen/Qwen3.5-4B"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=256 * 16 * 16, max_pixels=2048 * 2048
    )
    model = RBLNAutoModelForImageTextToText.from_pretrained(
        model_dir,
        export=False,
        rbln_config={
            "visual": {
                # The `device` parameter specifies which device should be used for each submodule during runtime.
                "device": [0, 1, 2, 3, 4, 5, 6, 7],
            },
            "device": [0, 1, 2, 3, 4, 5, 6, 7],
        },
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": load_image(IMAGE_URL)},
                {"type": "text", "text": "What animal is on the candy?"},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # autoregressively complete prompt
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(
        processor.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
    )


if __name__ == "__main__":
    main()
