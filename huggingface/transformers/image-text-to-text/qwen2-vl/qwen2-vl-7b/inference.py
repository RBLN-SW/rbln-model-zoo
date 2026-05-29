import os

from optimum.rbln import RBLNAutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


def main():
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(model_dir)
    model = RBLNAutoModelForVision2Seq.from_pretrained(
        model_dir,
        export=False,
        rbln_config={
            "visual": {
                # The `device` parameter specifies which device should be used for each submodule during runtime.
                "device": 0,
            },
            "device": [0, 1, 2, 3, 4, 5, 6, 7],
        },
    )

    # Messages containing a video url and a text query
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

    # In Qwen 2 VL, frame rate information is also input into the model to align with absolute time.
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        # Minimum and maximum pixel constraints for image or video processing, defined as patch counts.
        # Example: Set min_pixels and max_pixels to a patch range of 1024 to 5120 to balance performance and computational cost.
        # The max_pixels setting is closely tied to the visual model's max_seq_len, as it determines the maximum number of patches processed.
        min_pixels=1024
        * 14
        * 14,  # Minimum resolution in pixels (e.g., 1024 patches at patch size 14).
        max_pixels=5120
        * 14
        * 14,  # Maximum resolution in pixels (e.g., 5120 patches at patch size 14).
    )

    # autoregressively complete prompt
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    input_len = inputs.input_ids.shape[-1]
    generated_ids_trimmed = generated_ids[0][input_len:]

    # Show text and result
    print("--Result--")
    print(processor.decode(generated_ids_trimmed, skip_special_tokens=True))


if __name__ == "__main__":
    main()
