import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def generate_prompts(model_id: str, num_input_prompt: int):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train").shuffle(
        seed=42
    )

    prompts = []
    for i in range(num_input_prompt):
        image = dataset[i]["image"]
        question = dataset[i]["question"]

        # Use simple QA template because BLIP2 don't have default chat template.
        text_prompt = f"Question: {question}\n Answer:"

        prompts.append({"prompt": text_prompt, "multi_modal_data": {"image": [image]}})

    return prompts


def parse_args():
    parser = argparse.ArgumentParser()
    # You need to set the parameters following rbln_config.json in the compiled model
    parser.add_argument(
        "-m",
        "--model_id",
        dest="model_id",
        action="store",
        help="Compiled model directory path",
        default="Salesforce/blip2-opt-2.7b",
    )
    args = parser.parse_args()
    return args.model_id


def main():
    model_id = parse_args()
    llm = LLM(model=model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sampling_params = SamplingParams(
        temperature=0.0,
        skip_special_tokens=True,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=200,
    )
    inputs = generate_prompts(model_id, num_input_prompt)
    outputs = []

    for prompt in inputs:
        output = llm.generate(prompt, sampling_params=sampling_params)
        outputs.append(output)

    for i, output in enumerate(outputs):
        output = output[0].outputs[0].text
        print(f"Output {i}: {output}")


if __name__ == "__main__":
    main()
