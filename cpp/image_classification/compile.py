from argparse import ArgumentParser

import rebel
from torchvision import models


def parsing_argument() -> object:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name",
        "--model_name",
        dest="model_name",
        default="resnet18",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ],
        help="Model name",
    )
    return parser.parse_args()


def main() -> None:
    args = parsing_argument()

    weights = models.get_model_weights(args.model_name).DEFAULT
    model = getattr(models, args.model_name)(weights=weights).eval()

    compiled_model = rebel.compile_from_torch(
        model, [("x", [1, 3, 224, 224], "float32")]
    )
    compiled_model.save(f"{args.model_name}.rbln")


if __name__ == "__main__":
    main()
