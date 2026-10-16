import os

from optimum.rbln import RBLNAutoModelForMultimodalLM
from transformers import AutoProcessor


def main():
    model_id = "Qwen/Qwen3-ASR-1.7B-hf"

    # Load compiled model
    model = RBLNAutoModelForMultimodalLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    processor = AutoProcessor.from_pretrained(model_id)
    inputs = processor.apply_transcription_request(
        audio="https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
    ).to(model.device, model.dtype)

    # Generate with .generate()
    output_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]

    parsed = processor.decode(generated_ids, return_format="parsed")[0]
    print(f"Language: {parsed['language']}")
    print(f"Transcription: {parsed['transcription']}")


if __name__ == "__main__":
    main()
