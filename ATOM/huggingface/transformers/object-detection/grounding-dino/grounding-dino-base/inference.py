import os

import requests
from optimum.rbln import RBLNAutoModelForZeroShotObjectDetection
from PIL import Image
from transformers import AutoProcessor


def main():
    model_id = "IDEA-Research/grounding-dino-base"

    processor = AutoProcessor.from_pretrained(model_id, model_max_length=256)

    # Prepare inputs
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    text = "a cat. a remote control."

    inputs = processor(
        images=image,
        text=text,
        padding="max_length",
        do_pad=True,
        pad_size={"height": 1333, "width": 1333},
        return_tensors="pt",
    )

    # Load compiled model
    model = RBLNAutoModelForZeroShotObjectDetection.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate results
    output = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        output,
        inputs.input_ids,
        threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    # Result
    print(f"Result: {results}")


if __name__ == "__main__":
    main()
