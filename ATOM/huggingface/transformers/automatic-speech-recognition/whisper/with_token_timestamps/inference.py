import argparse
import os

import torch
from datasets import load_dataset
from optimum.rbln import RBLNAutoModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline


def validate_model_name(model_name: str) -> str:
    if model_name == "small":
        raise argparse.ArgumentTypeError(
            "Error: Choosing 'small' is currently disabled due to a known issue "
            "in the transformers library that may cause unexpected runtime errors. "
            "Please select an alternative model from the supported options."
        )
    return model_name


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=validate_model_name,
        choices=["tiny", "small", "base", "medium", "large-v3", "large-v3-turbo"],
        default="base",
        help="(str) model type, [tiny, small, base, medium, large-v3, large-v3-turbo]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"openai/whisper-{args.model_name}"

    # Prepare inputs
    processor = AutoProcessor.from_pretrained(model_id)
    dataset = load_dataset(
        "distil-whisper/librispeech_long", "clean", split="validation"
    )

    # Load compiled model
    model = RBLNAutoModelForSpeechSeq2Seq.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate with pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        return_timestamps="word",
        batch_size=1,
    )
    generate_kwargs = {
        "repetition_penalty": 1.3,
        "return_token_timestamps": True,
        "num_beams": 1,
    }

    with torch.no_grad():
        outputs = pipe(dataset[0]["audio"]["array"], generate_kwargs=generate_kwargs)

    print(f"Text: {outputs['text']}")
    print(f"Chunks: {outputs['chunks']}")


if __name__ == "__main__":
    main()
