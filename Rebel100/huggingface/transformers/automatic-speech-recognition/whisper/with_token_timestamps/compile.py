import argparse
import os

from optimum.rbln import RBLNAutoModelForSpeechSeq2Seq


def validate_model_name(model_name: str) -> str:
    if model_name == "small":
        raise argparse.ArgumentTypeError(
            "Error: Choosing 'small' is currently disabled due to a known issue "
            "in the transformers library that may cause unexpected runtime errors. "
            "Please select an alternative model from the supported options."
        )
    return model_name


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=validate_model_name,
        choices=["tiny", "small", "base", "medium", "large-v3", "large-v3-turbo"],
        default="base",
        help="(str) model type, [tiny, small, base, medium, large-v3, large-v3-turbo]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"openai/whisper-{args.model_name}"

    # Compile and export
    model = RBLNAutoModelForSpeechSeq2Seq.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_token_timestamps=True,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
