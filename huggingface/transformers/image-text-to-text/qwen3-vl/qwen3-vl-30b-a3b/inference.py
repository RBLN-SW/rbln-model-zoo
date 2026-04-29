import os

from optimum.rbln import RBLNAutoModelForVision2Seq
from transformers import AutoProcessor


def main():
    model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=256 * 16 * 16, max_pixels=2048 * 2048
    )
    model = RBLNAutoModelForVision2Seq.from_pretrained(
        model_dir,
        export=False,
        rbln_config={
            "visual": {
                # The `device` parameter specifies which device should be used for each submodule during runtime.
                "device": [0, 1, 2, 3, 4, 5, 6, 7],
            },
            "device": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        },
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # autoregressively complete prompt
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    print(
        processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


if __name__ == "__main__":
    main()
