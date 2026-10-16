import os

from optimum.rbln import RBLNAutoModelForImageTextToText


def main():
    model_id = "HuggingFaceM4/Idefics3-8B-Llama3"
    model = RBLNAutoModelForImageTextToText.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "text_model": {
                "batch_size": 1,
                "max_seq_len": 131_072,  # default "max_position_embeddings"
                "num_devices": 8,
                "use_inputs_embeds": True,
                "attn_impl": "flash_attn",
                "kvcache_partition_len": 16_384,  # Length of KV cache partitions for flash attention
            },
        },
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
