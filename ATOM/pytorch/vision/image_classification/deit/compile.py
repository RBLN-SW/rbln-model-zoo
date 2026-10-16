import argparse

import rebel  # RBLN Compiler
import timm
import torch

model_map = {
    "tiny": "deit_tiny_patch16_224",
    "small": "deit_small_patch16_224",
    "base": "deit_base_patch16_224",
    "tiny_distilled": "deit_tiny_distilled_patch16_224",
    "small_distilled": "deit_small_distilled_patch16_224",
    "base_distilled": "deit_base_distilled_patch16_224",
    "base_384": "deit_base_patch16_384",
    "base_distilled_384": "deit_base_distilled_patch16_384",
}


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="base",
        choices=[
            "tiny",
            "small",
            "base",
            "tiny_distilled",
            "small_distilled",
            "base_distilled",
            "base_384",
            "base_distilled_384",
        ],
        help="(str) type, DeiT model name. (tiny, small, base, tiny_distilled,"
        " small_distilled, base_distilled, base_384, base_distilled_384)",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name
    input_size = 384 if "384" in model_name else 224

    # Instantiate TorchVision model
    model = timm.create_model(f"{model_map[model_name]}.fb_in1k", pretrained=True)
    model.eval()

    # Compile torch model for ATOM
    input_info = [
        ("input_np", [1, 3, input_size, input_size], torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"{model_name}.rbln")


if __name__ == "__main__":
    main()
