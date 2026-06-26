import argparse

from datasets import load_dataset
from transformers import AutoProcessor

from vllm import LLM, SamplingParams


def generate_prompts(batch_size: int, model_id: str):
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="train").shuffle(seed=42)
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": dataset[i]["query"],
                    },
                ],
            },
        ]
        for i in range(batch_size)
    ]
    images = [[dataset[i]["image"]] for i in range(batch_size)]
    texts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = [
        {"prompt": text, "multi_modal_data": {"image": image}}
        for text, image in zip(texts, images)
    ]
    labels = [dataset[i]["label"] for i in range(batch_size)]
    return inputs, labels


def parse_args():
    parser = argparse.ArgumentParser()
    # You need to set the parameters following rbln_config.json in the compiled model
    parser.add_argument(
        "-m",
        "--model_id",
        dest="model_id",
        action="store",
        help="Compiled model directory path",
        default="mistral-community/pixtral-12b",
    )
    parser.add_argument(
        "-n",
        "--num-input_prompt",
        dest="num_input_prompt",
        type=int,
        action="store",
        help="The number of prompts",
        default=1,
    )
    return args.model_id, args.num_input_prompt


def main():
    model_id, num_input_prompt = parse_args()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    llm = LLM(model=model_id)

    inputs, labels = generate_prompts(num_input_prompt, model_id)

    outputs = []
    for prompt in inputs:
        output = llm.generate(prompt, sampling_params=sampling_params)
        outputs.append(output)

    for i, (output, label) in enumerate(zip(outputs, labels)):
        label_str = str(label)
        output = output[0].outputs[0].text

        print(f"Label {i}: {label_str}")
        print(f"Output {i}: {output}")


if __name__ == "__main__":
    main()
