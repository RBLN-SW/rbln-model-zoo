import os

from optimum.rbln import RBLNAutoModelForImageTextToText
from transformers import AutoProcessor


def main():
    model_id = "google/gemma-4-31B-it"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(model_id)
    model = RBLNAutoModelForImageTextToText.from_pretrained(model_dir, export=False)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
                },
                {"type": "text", "text": "What is shown in this image?"},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=512)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    print(processor.parse_response(response))


if __name__ == "__main__":
    main()
