import os

from optimum.rbln import RBLNAutoModelForImageTextToText


def main():
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    model = RBLNAutoModelForImageTextToText.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "language_model": {
                "num_devices": 1,
                "use_inputs_embeds": True,
                "max_seq_len": 32_768,  # default "max_position_embeddings"
                "attn_impl": "flash_attn",
                "kvcache_partition_len": 8_192,  # Length of KV cache partitions for flash attention
            }
        },
    )
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
