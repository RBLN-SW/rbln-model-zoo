import argparse
import urllib

import numpy as np
import rebel  # RBLN Runtime
import torch
import torchvision
from torchcodec.decoders import VideoDecoder


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

    # Prepare input video clip
    video_url = "https://github.com/pytorch/vision/raw/main/test/assets/videos/v_SoccerJuggling_g23_c01.avi"
    video_path = "./v_SoccerJuggling_g23_c01.avi"
    with urllib.request.urlopen(video_url) as response, open(video_path, "wb") as f:
        f.write(response.read())

    vid = VideoDecoder(video_path)[:16]
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    preprocess = weights.transforms()
    batch = preprocess(vid).unsqueeze(0)

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"{model_name}.rbln")

    # Run inference
    rebel_result = module.run(np.ascontiguousarray(batch, dtype=np.float32)).squeeze(0)
    rebel_result = np.exp(rebel_result) / np.sum(np.exp(rebel_result))

    # Display results
    label = torch.tensor(rebel_result).argmax().item()
    torch.tensor(rebel_result)[label].item()
    category_name = weights.meta["categories"][label]
    print(f"Top1 category: {category_name}")


if __name__ == "__main__":
    main()
