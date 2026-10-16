import argparse
import os
import sys
import urllib.request

import rebel
import torch

sys.path.append(os.path.join(sys.path[0], "MotionBERT"))
from lib.model.model_action import ActionNet
from lib.utils.learning import load_backbone
from lib.utils.tools import get_config

CONFIG = {
    "xsub": "/MotionBERT/configs/action/MB_train_NTU60_xsub.yaml",
    "xview": "/MotionBERT/configs/action/MB_train_NTU60_xview.yaml",
}

CHECKPOINT_FOLDER = {
    "xsub": "FT_MB_release_MB_ft_NTU60_xsub",
    "xview": "FT_MB_release_MB_ft_NTU60_xview",
}


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="xsub",
        choices=["xsub", "xview"],
        help="(str) type, motionbert action model_name.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    # prepare model
    cfg = get_config(sys.path[0] + CONFIG[model_name])

    MAX_CLIP_LEN = 243

    backbone = load_backbone(cfg)
    model = ActionNet(
        backbone=backbone,
        dim_rep=cfg.dim_rep,
        num_classes=cfg.action_classes,
        dropout_ratio=cfg.dropout_ratio,
        version=cfg.model_version,
        hidden_dim=cfg.hidden_dim,
        num_joints=cfg.num_joints,
    )

    save_path = "./checkpoint/action/" + CHECKPOINT_FOLDER[model_name] + "/"
    os.makedirs(save_path, exist_ok=True)
    hf_url = f"https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/action/{CHECKPOINT_FOLDER[model_name]}/best_epoch.bin"
    urllib.request.urlretrieve(hf_url, save_path + "best_epoch.bin")
    state_dict = torch.load(save_path + "best_epoch.bin")["model"]
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Compile torch model for ATOM
    input_info = [
        ("input_np", [1, 2, MAX_CLIP_LEN, 17, 3], torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"motionbert_action_{model_name}.rbln")


if __name__ == "__main__":
    main()
