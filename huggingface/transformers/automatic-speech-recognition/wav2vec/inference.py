import os

import torch
from datasets import load_dataset
from optimum.rbln import RBLNAutoModelForCTC
from transformers import AutoProcessor


def main():
    model_id = "facebook/wav2vec2-base-960h"

    processor = AutoProcessor.from_pretrained(model_id)

    # Prepare inputs
    ds = load_dataset(
        "patrickvonplaten/librispeech_asr_dummy",
        "clean",
        split="validation",
        trust_remote_code=True,
    )
    sample = ds[0]["audio"]
    input_values = processor(
        sample["array"],
        padding="max_length",
        max_length=160005,
        return_tensors="pt",
        truncation=True,
    ).input_values

    # Load compiled model
    model = RBLNAutoModelForCTC.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate transcription
    output = model(input_values)
    predicted_ids = torch.argmax(output.logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # Result
    print(f"Result: {transcription}")


if __name__ == "__main__":
    main()
