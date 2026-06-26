import os

from optimum.rbln import RBLNAutoModelForImageTextToText
from transformers import AutoProcessor
from transformers.image_utils import load_image


def main():
    model_id = "google/gemma-3-4b-it"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(model_id)
    model = RBLNAutoModelForImageTextToText.from_pretrained(model_dir, export=False)

    # Prepare image and text prompt, using the appropriate prompt template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What do we see in this image?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty.",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "And how about this image?"},
            ],
        },
    ]

    image1 = load_image(
        "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
    )
    image2 = load_image(
        "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg"
    )

    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)
    # Show text and result
    print(f"Result: {processor.decode(output[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
