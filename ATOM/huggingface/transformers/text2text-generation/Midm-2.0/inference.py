import argparse
import os

from optimum.rbln import RBLNAutoModelForCausalLM
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        choices=["Midm-2.0-Base-Instruct", "Midm-2.0-Mini-Instruct"],
        default="Midm-2.0-Base-Instruct",
        help="(str) model type, Size of midm. [Midm-2.0-Base-Instruct, Midm-2.0-Mini-Instruct]",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="KT에 대해 소개해줘",
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()

    model_id = f"K-intelligence/{args.model_id}"

    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    text = args.text
    messages = [
        {
            "role": "system",
            "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다.",
        },
        {"role": "user", "content": text},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")

    # Generate tokens
    output_sequence = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_length=32_768,
    )

    input_len = model_inputs.input_ids.shape[-1]
    generated_texts = tokenizer.decode(
        output_sequence[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Show text and result
    print(f"Text: {text}")
    print(f"Result: {generated_texts}")


if __name__ == "__main__":
    main()
