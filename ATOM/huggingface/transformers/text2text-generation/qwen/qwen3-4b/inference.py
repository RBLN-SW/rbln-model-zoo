import os

from optimum.rbln import RBLNAutoModelForCausalLM
from transformers import AutoTokenizer


def main():
    model_id = "Qwen/Qwen3-4B"

    # Load compiled model
    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt")

    # Generate tokens
    output_sequence = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_length=40960,
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
