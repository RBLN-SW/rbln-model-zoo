import os

from optimum.rbln import RBLNAutoModelForVision2Seq
from transformers import AutoProcessor
from transformers.image_utils import load_image


def main():
    model_id = "HuggingFaceM4/Idefics3-8B-Llama3"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(model_id)
    model = RBLNAutoModelForVision2Seq.from_pretrained(
        model_dir,
        export=False,
        rbln_config={
            # The `device` parameter specifies the device allocation for each submodule during runtime.
            # As Idefics3 consists of multiple submodules, loading them all onto a single device may exceed its memory capacity, especially as the batch size increases.
            # By distributing submodules across devices, memory usage can be optimized for efficient runtime performance.
            "vision_model": {
                "device": 0,
            },
            "text_model": {
                "device": [0, 1, 2, 3, 4, 5, 6, 7],
            },
        },
    )

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

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
    inputs = {k: v for k, v in inputs.items()}

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    # Show text and result
    print(processor.batch_decode(output, skip_special_tokens=True))


if __name__ == "__main__":
    main()
