import argparse
import os
import sys
import urllib.request

import imageio
import numpy as np
import rebel
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(sys.path[0], "MotionBERT"))
from lib.data.dataset_wild import WildDetDataset
from lib.utils.tools import get_config
from lib.utils.utils_data import flip_data
from lib.utils.vismo import render_and_save

CONFIG = {
    "base": "/MotionBERT/configs/pose3d/MB_ft_h36m.yaml",
    "lite": "/MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml",
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

    # ref: https://github.com/KevinLTT/video2bvh/blob/master/miscs/cxk.mp4
    vid_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/dataset/motionbert/sample.mp4"
    urllib.request.urlretrieve(vid_url, "./sample.mp4")
    json_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/dataset/motionbert/alphapose-results.json"
    urllib.request.urlretrieve(json_url, "./alphapose-results.json")

    testloader_params = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
        "drop_last": False,
    }
    vid = imageio.get_reader("./sample.mp4", "ffmpeg")
    fps_in = vid.get_meta_data()["fps"]
    wild_dataset = WildDetDataset(
        "alphapose-results.json", clip_len=MAX_CLIP_LEN, scale_range=[1, 1], focus=None
    )
    test_loader = DataLoader(wild_dataset, **testloader_params)

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"motionbert_pose3d_{model_name}.rbln", tensor_type="pt")

    # inference
    results_all = []
    for batch_input in test_loader:
        T = batch_input.shape[1]
        if cfg.no_conf:
            batch_input = batch_input[:, :, :, :2]
        if T != MAX_CLIP_LEN:
            repeat_end_of_frames = [batch_input] + [
                batch_input[:, -1:, :, :] for _ in range(MAX_CLIP_LEN - T)
            ]
            batch_input = torch.concat(repeat_end_of_frames, dim=1)
        if cfg.flip:
            batch_input_flip = flip_data(batch_input)
            predicted_3d_pos_1 = module.run(batch_input)
            predicted_3d_pos_flip = module.run(batch_input_flip)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
            predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
        else:
            predicted_3d_pos = module.run(batch_input)
        if cfg.rootrel:
            predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
        else:
            predicted_3d_pos[:, 0, 0, 2] = 0
            pass
        if cfg.gt_2d:
            predicted_3d_pos[..., :2] = batch_input[..., :2]
        if T != MAX_CLIP_LEN:
            predicted_3d_pos = predicted_3d_pos[:, :T, :, :]
        results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)

    # rendering video using result
    render_and_save(
        results_all, f"./3dpose_{model_name}.mp4", keep_imgs=False, fps=fps_in
    )


if __name__ == "__main__":
    main()
