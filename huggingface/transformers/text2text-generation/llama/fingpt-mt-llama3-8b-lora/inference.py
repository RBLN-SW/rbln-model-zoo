import os

from optimum.rbln import RBLNAutoModelForCausalLM
from transformers import AutoTokenizer


def main():
    model_id = "meta-llama/Meta-Llama-3-8B"
    lora_id = "FinGPT/fingpt-mt_llama3-8b_lora"
    prompt = """Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: Tech Startup ASDF AI Secures $50M Series B Funding
Emerging artificial intelligence company ASDF AI has successfully raised $50 million in Series B funding, led by venture capital firm ZXCV Ventures. 
The startup, known for its innovative natural language processing solutions, plans to expand its workforce and accelerate product development. 
CEO John Doe expressed optimism about the company's growth trajectory in the competitive AI market.
Answer: """  # noqa: E501

    # Load compiled model
    model_save_dir = f"{os.path.basename(model_id)}_{os.path.basename(lora_id)}"
    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=model_save_dir, export=False
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate tokens
    output_sequence = model.generate(**inputs, max_length=512)
    input_len = inputs.input_ids.shape[-1]
    generated_texts = tokenizer.decode(
        output_sequence[0][input_len:], skip_special_tokens=True
    )

    # Show text and result
    print(f"Text: {prompt}")
    print(f"Result: {generated_texts}")


if __name__ == "__main__":
    main()
