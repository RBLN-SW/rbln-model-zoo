import argparse

import requests
from PIL import Image
from transformers import LlavaNextProcessor

from vllm import LLM, SamplingParams

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_id",
        dest="model_id",
        action="store",
        help="Compiled model directory path",
        default="llava-hf/llava-v1.6-mistral-7b-hf",
    )
    args = parser.parse_args()
    return args.model_id


def main():
    model_id = parse_args()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    processor = LlavaNextProcessor.from_pretrained(model_id)

    llm = LLM(
        model=model_id,
        limit_mm_per_prompt={"image": 1},  # The maximum number to accept
    )

    url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = []
    for _ in range(10):
        data = {"prompt": prompt, "multi_modal_data": {"image": image}}
        inputs.append(data)

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for i, o in enumerate(outputs):
        generated_text = o.outputs[0].text
        print(f"Output {i}: {generated_text}")


if __name__ == "__main__":
    main()
