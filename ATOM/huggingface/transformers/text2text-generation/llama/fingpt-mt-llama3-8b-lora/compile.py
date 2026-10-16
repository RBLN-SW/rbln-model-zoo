import os

from optimum.rbln import RBLNAutoModelForCausalLM
from peft import PeftModel
from transformers import LlamaForCausalLM


def main():
    model_id = "meta-llama/Meta-Llama-3-8B"
    lora_id = "FinGPT/fingpt-mt_llama3-8b_lora"

    # Load the base Llama model
    model = LlamaForCausalLM.from_pretrained(model_id)

    # Apply LoRA adapter to the base model
    model = PeftModel.from_pretrained(model, lora_id)

    # Merge LoRA weights into the base model:
    # 1. Combines the LoRA adapter's trained weights with the base model parameters
    # 2. This creates a single, standalone model that includes all LoRA adaptations
    # 3. After merging, the separate LoRA weights are no longer needed and are unloaded
    # 4. This reduces memory usage since we don't need to keep both base and adapter weights
    # 5. The resulting model can be used without the PEFT library as it's now a standard model
    model = model.merge_and_unload()

    # Convert to RBLN-optimized model format
    model = RBLNAutoModelForCausalLM.from_model(
        model,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=8192,  # default "max_position_embeddings"
        rbln_num_devices=4,
    )

    # Save the optimized model using the base model name as directory
    model_save_dir = f"{os.path.basename(model_id)}_{os.path.basename(lora_id)}"
    model.save_pretrained(model_save_dir)


if __name__ == "__main__":
    main()
