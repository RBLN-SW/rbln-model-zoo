import argparse
import os
import sys
import urllib.request

import rebel
from torch.utils.data import DataLoader

sys.path.append(os.path.join(sys.path[0], "MotionBERT"))
from lib.data.dataset_action import NTURGBD
from lib.utils.tools import get_config

CONFIG = {
    "xsub": "/MotionBERT/configs/action/MB_train_NTU60_xsub.yaml",
    "xview": "/MotionBERT/configs/action/MB_train_NTU60_xview.yaml",
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

    # prepare dataset
    cfg = get_config(sys.path[0] + CONFIG[model_name])

    os.makedirs("./data/action", exist_ok=True)
    vid_url = (
        "https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl"
    )
    urllib.request.urlretrieve(vid_url, "./data/action/ntu60_hrnet.pkl")

    testloader_params = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
    }
    data_path = "./data/action/%s.pkl" % cfg.dataset
    ntu60_dataset = NTURGBD(
        data_path=data_path,
        data_split=cfg.data_split + "_val",
        n_frames=cfg.clip_len,
        random_move=False,
        scale_range=cfg.scale_range_test,
    )
    test_dataloader = DataLoader(ntu60_dataset, **testloader_params)
    batch_input, _ = next(iter(test_dataloader))

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"motionbert_action_{model_name}.rbln", tensor_type="pt")

    # Run model
    out = module.run(batch_input)

    # Get TOP5 classify result
    _, pred = out.topk(5, 1, True, True)
    pred.t()
    pred = pred[0]

    # show result
    print(f"Top5: {pred.tolist()}")
    print(f"Top1: {pred[0].item()}")


if __name__ == "__main__":
    main()
