import argparse

import rebel  # RBLN Compiler
import torch
import torchvision


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="r3d_18",
        choices=[
            "s3d",
            "r3d_18",
            "mc3_18",
            "r2plus1d_18",
        ],
        help="(str) type, torchvision model name. (s3d, r3d_18, mc3_18, r2plus1d_18)",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    # Instantiate TorchVision model
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    model = getattr(torchvision.models.video, model_name)(weights=weights).eval()

    # Get input shape
    input_np = torch.rand(16, 3, 240, 320)
    input_np = weights.transforms()(input_np).unsqueeze(0)

    # Compile torch model for ATOM
    input_info = [
        ("input_np", list(input_np.shape), torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"{model_name}.rbln")


if __name__ == "__main__":
    main()
