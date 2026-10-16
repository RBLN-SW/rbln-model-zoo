from optimum.rbln import RBLNLlamaForCausalLM

# Export huggingFace pytorch llama3 model to RBLN compiled model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
compiled_model = RBLNLlamaForCausalLM.from_pretrained(
    model_id=model_id,
    export=True,
    rbln_max_seq_len=8192,
    rbln_num_devices=4,
    rbln_batch_size=4,  # batch_size > 1 is recommended for continuous batching
)

compiled_model.save_pretrained("rbln-Llama-3-8B-Instruct")
