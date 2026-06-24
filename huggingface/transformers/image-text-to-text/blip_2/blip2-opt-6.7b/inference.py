import os

import requests
from optimum.rbln import RBLNAutoModelForImageTextToText
from PIL import Image
from transformers import AutoProcessor


def main():
    model_id = "Salesforce/blip2-opt-6.7b"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(model_id)
    model = RBLNAutoModelForImageTextToText.from_pretrained(model_dir, export=False)

    # Prepare image and text prompt, using the appropriate prompt template
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    prompt = "Question: how many cats are there? Answer:"

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v for k, v in inputs.items()}

    # autoregressively complete prompt
    output = model.generate(**inputs)

    # Show text and result
    print(
        f"Result: {processor.batch_decode(output, skip_special_tokens=True)[0].strip()}"
    )


if __name__ == "__main__":
    main()
