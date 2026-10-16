import os
import sys

sys.path.append(os.path.join(sys.path[0], "Depth-Anything-3", "src"))

import argparse

from modeling_depth_anything_3 import RBLNDepthAnything3, RBLNDepthAnything3Config
from optimum.rbln import RBLNAutoConfig, RBLNAutoModel

RBLNAutoConfig.register(RBLNDepthAnything3Config)
RBLNAutoModel.register(RBLNDepthAnything3)


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="depth-anything/DA3-BASE",
        choices=[
            "depth-anything/DA3-SMALL",
            "depth-anything/DA3-BASE",
            "depth-anything/DA3-LARGE",
            "depth-anything/DA3-GIANT",
            "depth-anything/DA3-LARGE-1.1",
            "depth-anything/DA3-GIANT-1.1",
        ],
        help="(str) Hugging Face model id.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = args.model_id

    # Compile and export.
    # Args:
    #   - rbln_use_ray_pose: Pass True when inference uses use_ray_pose=True (default False).
    model = RBLNDepthAnything3.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_num_images=2,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
