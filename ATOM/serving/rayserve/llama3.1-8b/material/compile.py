import os

from optimum.rbln import RBLNLlamaForCausalLM

model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Compile and export
model = RBLNLlamaForCausalLM.from_pretrained(
    model_id=model_id,
    export=True,  # Export a PyTorch model to RBLN model with Optimum
    rbln_batch_size=1,  # Batch size
    rbln_max_seq_len=131_072,  # Maximum sequence length
    rbln_num_devices=8,
    rbln_attn_impl="flash_attn",  # Use Flash Attention
    rbln_kvcache_partition_len=16_384,  # Length of KV cache partitions for flash attention
)

# Save compiled results to disk
model.save_pretrained(os.path.basename(model_id))
