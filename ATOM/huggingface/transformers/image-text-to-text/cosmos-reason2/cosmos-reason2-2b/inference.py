import os

from optimum.rbln import RBLNAutoModelForImageTextToText
from transformers import AutoProcessor


def main():
    model_id = "nvidia/Cosmos-Reason2-2B"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(model_dir)
    model = RBLNAutoModelForImageTextToText.from_pretrained(
        model_dir,
        export=False,
        rbln_config={
            "visual": {
                # The `device` parameter specifies which device should be used for each submodule during runtime.
                "device": [0, 1, 2, 3, 4, 5, 6, 7],
            },
            "device": [0, 1, 2, 3, 4, 5, 6, 7],
        },
    )

    video_messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": "https://raw.githubusercontent.com/nvidia-cosmos/cosmos-reason1/main/assets/sample.mp4",
                    "fps": 4,
                },
                {
                    "type": "text",
                    "text": (
                        "Is it safe to turn right? Answer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag."
                    ),
                },
            ],
        },
    ]

    # Process inputs
    inputs = processor.apply_chat_template(
        video_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=4,
    )

    # autoregressively complete prompt
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    # Show text and result
    print(f"Result: {output_text}")


if __name__ == "__main__":
    main()
