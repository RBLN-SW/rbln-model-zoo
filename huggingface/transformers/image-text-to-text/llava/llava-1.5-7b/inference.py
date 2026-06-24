import os

import requests
from optimum.rbln import RBLNAutoModelForImageTextToText
from PIL import Image
from transformers import AutoProcessor


def main():
    model_id = "llava-hf/llava-1.5-7b-hf"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(model_id)
    model = RBLNAutoModelForImageTextToText.from_pretrained(model_dir, export=False)

    # Prepare image and text prompt, using the appropriate prompt template
    url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    # Show text and result
    print(f"Result: {processor.decode(output[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
