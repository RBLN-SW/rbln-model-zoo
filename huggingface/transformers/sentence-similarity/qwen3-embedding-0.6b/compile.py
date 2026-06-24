import os

from optimum.rbln import RBLNAutoModel


def main():
    model_id = "Qwen/Qwen3-Embedding-0.6B"

    # Compile and export
    model = RBLNAutoModel.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=32_768,  # default "max_position_embeddings"
        rbln_num_devices=1,
        rbln_attn_impl="flash_attn",
        rbln_kvcache_partition_len=16_384,  # Length of KV cache partitions for flash attention
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
