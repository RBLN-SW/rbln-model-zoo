import os

from optimum.rbln import RBLNAutoModelForImageTextToText
from transformers import AutoProcessor
from transformers.image_utils import load_image


def main():
    model_id = "google/paligemma2-3b-mix-224"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(model_id)
    model = RBLNAutoModelForImageTextToText.from_pretrained(model_dir, export=False)

    # Prepare image and text prompt, using the appropriate prompt template
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    image = load_image(url)

    prompt = "describe en"
    model_inputs = processor(text=prompt, images=image, return_tensors="pt")
    input_len = model_inputs["input_ids"].shape[-1]

    # autoregressively complete prompt
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)

    # Show text and result
    print(f"Result: {decoded}")


if __name__ == "__main__":
    main()
