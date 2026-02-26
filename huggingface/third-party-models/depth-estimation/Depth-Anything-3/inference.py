import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(sys.path[0], "Depth-Anything-3", "src"))

import imageio
import numpy as np
from depth_anything_3.utils.visualize import visualize_depth
from modeling_depth_anything_3 import RBLNDepthAnything3, RBLNDepthAnything3Config
from optimum.rbln import RBLNAutoConfig, RBLNAutoModel
from transformers import PretrainedConfig

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

    # Load compiled model
    args = parsing_argument()
    model_id = args.model_id
    model = RBLNDepthAnything3.from_pretrained(
        os.path.basename(model_id),
        export=False,
        config=PretrainedConfig(model_type="depth_anything"),
    )

    # Load the image (SOH multi-view: https://github.com/ByteDance-Seed/Depth-Anything-3/tree/main/assets/examples/SOH)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_path = os.path.join(
        script_dir, "Depth-Anything-3", "assets", "examples", "SOH"
    )
    images = sorted(glob.glob(os.path.join(example_path, "*.png")))

    # Inference.
    # Args:
    #   - use_ray_pose: Pass True for pose from ray (default False: pose from cam_dec when compiled with rbln_use_ray_pose=False).
    prediction = model.inference(images)

    print(prediction.depth.shape)
    print(prediction.conf.shape)
    print(prediction.extrinsics.shape)
    print(prediction.intrinsics.shape)

    # Optional: print pose statistics
    mean_std = lambda x: f"{float(np.mean(x)):.6f} / {float(np.std(x)):.6f}"
    print(f"extrinsics mean/std: {mean_std(prediction.extrinsics)}")
    print(f"intrinsics mean/std: {mean_std(prediction.intrinsics)}")

    # Optional: save depth/conf images
    out_dir = Path(script_dir) / "output"
    out_dir.mkdir(exist_ok=True)
    depth = np.asarray(prediction.depth)
    conf = np.asarray(prediction.conf) if prediction.conf is not None else None
    depth_maps = depth.reshape((-1,) + depth.shape[-2:])
    conf_maps = conf.reshape((-1,) + conf.shape[-2:]) if conf is not None else None
    items = (
        zip(depth_maps, conf_maps)
        if conf_maps is not None
        else ((d, None) for d in depth_maps)
    )
    for i, (depth_map, conf_map) in enumerate(items):
        imageio.imwrite(out_dir / f"depth_{i:04d}.png", visualize_depth(depth_map))
        if conf_map is None:
            continue
        p2, p98 = np.percentile(conf_map, [2, 98])
        vis = np.clip((conf_map - p2) / (p98 - p2 + 1e-8), 0, 1)
        imageio.imwrite(out_dir / f"conf_{i:04d}.png", (vis * 255).astype(np.uint8))

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
