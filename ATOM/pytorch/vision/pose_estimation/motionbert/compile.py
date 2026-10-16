import argparse
import os
import sys
import urllib.request

import rebel
import torch

sys.path.append(os.path.join(sys.path[0], "MotionBERT"))
from lib.utils.learning import load_backbone
from lib.utils.tools import get_config

CONFIG = {
    "base": "/MotionBERT/configs/pose3d/MB_ft_h36m.yaml",
    "lite": "/MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml",
}

CHECKPOINT_FOLDER = {
    "base": "FT_MB_release_MB_ft_h36m",
    "lite": "FT_MB_lite_MB_ft_h36m_global_lite",
}


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="base",
        choices=["base", "lite"],
        help="(str) type, motionBERT 3D pose model name.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    # prepare model
    cfg = get_config(sys.path[0] + CONFIG[model_name])

    MAX_CLIP_LEN = 243

    model = load_backbone(cfg)

    save_path = "./checkpoint/pose3d/" + CHECKPOINT_FOLDER[model_name] + "/"
    os.makedirs(save_path, exist_ok=True)
    hf_url = f"https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d/{CHECKPOINT_FOLDER[model_name]}/best_epoch.bin"
    urllib.request.urlretrieve(hf_url, save_path + "best_epoch.bin")
    state_dict = torch.load(save_path + "best_epoch.bin")["model_pos"]
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Compile torch model for ATOM
    input_info = [
        ("input_np", [1, MAX_CLIP_LEN, 17, 3], torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"motionbert_pose3d_{model_name}.rbln")


if __name__ == "__main__":
    main()
