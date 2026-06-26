import os

import torch
from optimum.rbln import RBLNAutoModelForImageTextToText
from transformers import AutoProcessor


def main():
    model_id = "LGAI-EXAONE/EXAONE-4.5-33B"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(model_dir)
    model = RBLNAutoModelForImageTextToText.from_pretrained(
        model_dir,
        export=False,
    )

    # Messages containing a video url and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                },
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    # autoregressively complete prompt
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    input_len = inputs["input_ids"].shape[-1]
    generated_ids_trimmed = generated_ids[0][input_len:]

    # Show text and result
    print(
        f"Result: {processor.decode(generated_ids_trimmed, skip_special_tokens=True)}"
    )


if __name__ == "__main__":
    main()
