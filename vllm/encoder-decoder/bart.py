import argparse
import asyncio
import sys

from transformers import AutoTokenizer

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


def initialize(model_dir):
    engine_args = AsyncEngineArgs(
        model=model_dir,
    )

    return engine_args


def stop_tokens(tokenizer):
    eot_id = next(
        (
            k
            for k, t in tokenizer.added_tokens_decoder.items()
            if t.content == "<|eot_id|>"
        ),
        None,
    )
    if eot_id is not None:
        return [tokenizer.eos_token_id, eot_id]
    else:
        return [tokenizer.eos_token_id]


async def run_single(engine, sampling_params, chat, request_id):
    results_generator = engine.generate(
        chat, sampling_params, request_id=str(request_id)
    )
    final_result = None
    async for result in results_generator:
        # You can use the intermediate `result` here, if needed.
        final_result = result
    return final_result


async def run_multi(engine, sampling_params, prompts):
    tasks = [
        asyncio.create_task(run_single(engine, sampling_params, prompt, i))
        for (i, prompt) in enumerate(prompts)
    ]
    return [await task for task in tasks]


def parse_args():
    parser = argparse.ArgumentParser()

    # [info] `facebook/bart-base` and `facebook/bart-large` are
    # primarily designed for feature extraction.
    # For text-to-text generation (fill-mask task) in vLLM,
    # it is recommended to use the BartForConditionalGeneration model
    # (ex. `lucadiliello/bart-small` model).
    parser.add_argument(
        "-m",
        "--model_dir",
        dest="model_dir",
        action="store",
        help="Model directory path",
        default="lucadiliello/bart-small",
    )
    args = parser.parse_args()
    return args.model_dir


def main():
    model_dir = parse_args()
    engine_args = initialize(model_dir)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=False,
        skip_special_tokens=True,
        stop_token_ids=stop_tokens(tokenizer),
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Runs a single inference for an example
    INPUT_PROMPT = "UN Chief Says There Is No <mask> in Syria"
    result = asyncio.run(run_single(engine, sampling_params, INPUT_PROMPT, "123"))
    print(f"Result: {result}")

    # Runs multi inference for an example
    INPUT_PROMPTS = [
        "The capital city of France is <mask>.",
        "She opened the door to find a <mask> waiting on the other side.",
        "The <mask> was so <mask> that everyone started clapping.",
        "There are <mask> continents on Earth.",
        "<mask> is the largest planet in our solar system.",
        "To bake a cake, you need flour, sugar, eggs, and <mask>.",
        "He <mask> the ball over the fence and scored a home run.",
        "After hearing the news, she felt <mask> and started crying.",
        "<mask> is known for his theory of relativity.",
        "The Great Wall of China is located in <mask>.",
    ]
    results = asyncio.run(run_multi(engine, sampling_params, INPUT_PROMPTS))
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
