import os
import sys

sys.path.insert(0, os.path.join(sys.path[0], "colpali"))

import torch
from colpali.colpali_engine.models import ColPali
from optimum.rbln import RBLNColPaliForRetrieval
from peft.tuners import lora
from transformers import ColPaliConfig, ColPaliForRetrieval
from transformers.modeling_utils import no_init_weights


# Merge and unload LoRA layers
def merge_and_unload_lora(module):
    for name, child in module.named_children():
        if isinstance(child, lora.Linear) and isinstance(child, torch.nn.Module):
            lora.Linear.merge(child)
            setattr(module, name, child.base_layer)
        else:
            # Recurse into child modules
            merge_and_unload_lora(child)


# Convert ColPali model to Hugging Face ColPaliForRetrieval model
def convert_colpali_to_hf_class(model: ColPali):
    config = ColPaliConfig(
        vlm_config=model.model.config,
        embedding_dim=model.dim,
    )
    with no_init_weights():
        hf_model = ColPaliForRetrieval(config=config).to("cpu").eval()
    hf_model.vlm = model.model
    hf_model.embedding_proj_layer = model.custom_text_proj

    return hf_model


def main():
    model_id = "vidore/colpali-v1.3"

    # Load model from ColPaliEngine
    model = ColPali.from_pretrained(model_id).eval()
    merge_and_unload_lora(model)
    model = convert_colpali_to_hf_class(model)

    # Compile and export
    rbln_model = RBLNColPaliForRetrieval.from_model(
        model,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=2,
        rbln_tensor_parallel_size=4,
        rbln_config={
            "vlm": {
                "language_model": {
                    # For the definition of prefill_chunk_size, please refer to the link below.
                    # https://docs.rbln.ai/v0.9.4/software/optimum/model_api/transformers/text_model/common.html#optimum.rbln.transformers.models.decoderonly.configuration_decoderonly.RBLNDecoderOnlyModelForCausalLMConfig.__init__
                    # default "max_position_embeddings"
                    "prefill_chunk_size": 8192,
                },
            }
        },
    )

    # Save compiled results to disk
    rbln_model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
