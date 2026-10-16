import argparse
import urllib

import rebel  # noqa: F401  # needed to use torch dynamo's "rbln" backend.
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

    # Instantiate TorchVision model
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    model = getattr(torchvision.models.video, model_name)(weights=weights).eval()

    # Prepare input video clip
    video_url = "https://github.com/pytorch/vision/raw/main/test/assets/videos/v_SoccerJuggling_g23_c01.avi"
    video_path = "./v_SoccerJuggling_g23_c01.avi"
    with urllib.request.urlopen(video_url) as response, open(video_path, "wb") as f:
        f.write(response.read())

    vid = VideoDecoder(video_path)[:16]
    preprocess = weights.transforms()
    batch = preprocess(vid).unsqueeze(0).contiguous()

    # Compile the model
    model = torch.compile(
        model,
        backend="rbln",
        # Disable dynamic shape support, as the RBLN backend currently does not support it
        dynamic=False,
        options={"cache_dir": f"./{model_name}"},
    )

    # (Optional) First call of forward invokes the compilation
    model(batch)

    # After that, You can use models compiled for RBLN hardware
    rbln_result = model(batch)

    # Process results
    rbln_result = torch.softmax(rbln_result, dim=-1)

    # Display results
    label = rbln_result.argmax().item()
    confidence = rbln_result[0, label].item()
    category_name = weights.meta["categories"][label]
    print(f"Top1 category: {category_name}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
