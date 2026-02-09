import os

from optimum.rbln import RBLNAutoModelForCausalLM
from transformers import AutoTokenizer


def main():
    """
    EXAONE Model Usage License:

    - Solely for research purposes. This includes evaluation, testing, academic research, experimentation,
      and participation in competitions, provided that such participation is in a non-commercial context.
    - For commercial use and larger context length, please contact LG AI Research, contact_us@lgresearch.ai
    - Please refer to License Policy for detailed terms and conditions: https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct/blob/main/LICENSE
    """  # noqa: E501
    pinned_revision = "e949c91dec92095908d34e6b560af77dd0c993f8"

    model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    # Load compiled model
    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=pinned_revision,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    text = "Explain who you are"
    messages = [
        {
            "role": "system",
            "content": "You are EXAONE model from LG AI Research, a helpful assistant.",
        },
        {"role": "user", "content": text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Generate tokens
    output_sequence = model.generate(input_ids, max_length=32768)

    input_len = input_ids.shape[-1]
    generated_texts = tokenizer.decode(
        output_sequence[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Show text and result
    print("--- Text ---")
    print(text)
    print("--- Result ---")
    print(generated_texts)


if __name__ == "__main__":
    main()
