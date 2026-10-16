import argparse
import os

from optimum.rbln import RBLNAutoModelForSequenceClassification

MAX_SEQ_LEN_CFG = {
    "v2-m3": 8192,
    "large": 512,
    "base": 512,
}


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["v2-m3", "large", "base"],
        default="v2-m3",
        help="(str) model type, Size of bge-reranker. [v2-m3, large, base]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"BAAI/bge-reranker-{args.model_name}"

    # Compile and export
    model = RBLNAutoModelForSequenceClassification.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=MAX_SEQ_LEN_CFG[
            args.model_name
        ],  # default "max_position_embeddings"
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
