import argparse
import os

from optimum.rbln import RBLNAutoModelForCausalLM
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        choices=["A.X-4.0-Light"],
        default="A.X-4.0-Light",
        help="(str) model type, Size of A.X-4.0. [A.X-4.0-Light]",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The first human went into space and orbited the Earth on April 12, 1961.",
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"skt/{args.model_id}"

    # Load compiled model
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
            "content": "당신은 사용자가 제공하는 영어 문장들을 한국어로 번역하는 AI 전문가입니다.",
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
        max_length=16_384,
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
