import os

from optimum.rbln import RBLNAutoModelForImageTextToText
from transformers import AutoProcessor


def main():
    model_id = "mistral-community/pixtral-12b"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(model_id)
    model = RBLNAutoModelForImageTextToText.from_pretrained(model_dir, export=False)

    # Prepare image and text prompt, using the appropriate prompt template
    IMG_URLS = [
        "https://picsum.photos/id/237/400/300",
    ]
    PROMPT = "<s>[INST]Describe the images.\n[IMG][/INST]"
    inputs = processor(text=PROMPT, images=IMG_URLS, return_tensors="pt")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=500)

    # Show text and result
    print(
        processor.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    )


if __name__ == "__main__":
    main()
