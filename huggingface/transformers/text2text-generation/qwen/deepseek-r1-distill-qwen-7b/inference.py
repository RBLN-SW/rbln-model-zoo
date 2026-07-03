import argparse
import os

from optimum.rbln import RBLNAutoModelForCausalLM
from tokenizers import decoders
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        default="Give me a short introduction to large language model.",
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # Load compiled model
    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # Workaround: transformers v5 replaces the ByteLevel pre-tokenizer /
    # decoder in DeepSeek-R1-Distill fast tokenizers with Metaspace, so
    # `decode()` cannot reverse byte-level BPE and emits raw `Ġ` / `Ċ`
    # markers. Reattach a ByteLevel decoder explicitly until the upstream
    # fix lands. Ref: https://github.com/huggingface/transformers/issues/45488
    tokenizer.backend_tokenizer.decoder = decoders.ByteLevel()
    conversation = [{"role": "user", "content": args.text}]
    text = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Generate tokens
    output_sequence = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=131_072,
    )

    input_len = inputs.input_ids.shape[-1]
    generated_texts = tokenizer.decode(
        output_sequence[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Show text and result
    print(f"Text: {args.text}")
    print(f"Result: {generated_texts}")


if __name__ == "__main__":
    main()
