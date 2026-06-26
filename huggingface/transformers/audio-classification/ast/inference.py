import os

import torch
from datasets import load_dataset
from optimum.rbln import RBLNAutoModelForAudioClassification
from transformers import AutoFeatureExtractor


def main():
    model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    # Prepare inputs
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_demo",
        "clean",
        split="validation",
        trust_remote_code=True,
    )
    sample = ds[0]["audio"]
    input_values = feature_extractor(sample["array"], return_tensors="pt").input_values

    # Load compiled model
    model = RBLNAutoModelForAudioClassification.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Classify audio
    output = model(input_values)
    class_ids = torch.argmax(output[0], dim=-1).item()
    label = model.config.id2label[class_ids]

    # Result
    print(f"Result: {label}")


if __name__ == "__main__":
    main()
