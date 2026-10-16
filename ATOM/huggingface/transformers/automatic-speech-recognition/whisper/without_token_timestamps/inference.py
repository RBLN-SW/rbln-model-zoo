import argparse
import os

from datasets import load_dataset
from optimum.rbln import RBLNAutoModelForSpeechSeq2Seq
from transformers import AutoProcessor


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["tiny", "small", "base", "medium", "large-v3", "large-v3-turbo"],
        default="base",
        help="(str) model type, [tiny, small, base, medium, large-v3, large-v3-turbo]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"openai/whisper-{args.model_name}"

    # Prepare long form inputs
    processor = AutoProcessor.from_pretrained(model_id)
    ds = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

    input = processor(
        ds[0]["audio"]["array"],
        sampling_rate=ds[0]["audio"]["sampling_rate"],
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Load compiled model
    model = RBLNAutoModelForSpeechSeq2Seq.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate with .generate()
    outputs = model.generate(**input, return_timestamps=True)
    generated_ids = outputs
    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for i, transcription in enumerate(transcriptions):
        print(f"Transcription {i}: {transcription}")


if __name__ == "__main__":
    main()
