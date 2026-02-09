import os

from optimum.rbln import RBLNAutoModelForCausalLM
from transformers import AutoConfig


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

    confg = AutoConfig.from_pretrained(
        model_id,
        revision=pinned_revision,
        trust_remote_code=True,
    )

    # Compile and export
    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=model_id,
        revision=pinned_revision,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=32768,  # default "max_position_embeddings"
        rbln_tensor_parallel_size=4,
        config=confg,
        trust_remote_code=True,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
