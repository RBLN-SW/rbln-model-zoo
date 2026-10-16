import argparse
import os
import sys
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image

sys.path.insert(0, os.path.join(sys.path[0], "sam2"))

import rebel  # noqa: F401  # needed to use torch dynamo's "rbln" backend
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CFG_CONFIG = {
    "sam2.1_hiera_large": {
        "config_file": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "sam2.1_hiera_large.pt",
    },
}


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "sam2.1_hiera_large",
        ],
        default="sam2.1_hiera_large",
        help="Model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        choices=[1, 2],
        default=1,
        help="Batch size.",
    )
    return parser.parse_args()


def run_download_script():
    print("Downloading model checkpoint script...")
    try:
        url = "https://raw.githubusercontent.com/facebookresearch/sam2/main/checkpoints/download_ckpts.sh"
        script_path = "download_ckpts.sh"
        with open(script_path, "wb") as file:
            file.write(requests.get(url).content)

        # Run the downloaded script
        os.system(f"bash {script_path}")

        print("File downloaded successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def download_file(url, output_path):
    print(f"Downloading file from {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print("File downloaded successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


class WrapperSAM2ImagePredictor(SAM2ImagePredictor):
    def set_image(self, image: Union[np.ndarray, Image.Image]) -> None:
        batch_size = image.shape[0]

        backbone_out = self.model.forward_image(image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        vision_feats[-1] += self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        return feats[-1], feats[:-1]

    def set_image_preprocess(self, image_list):
        self.reset_predictor()
        self._orig_hw = []
        for image in image_list:
            self._orig_hw.append(image.shape[:2])

        img_batch = [self._transforms(img) for img in image_list]
        img_batch = torch.stack(img_batch, dim=0)

        return img_batch

    def set_image_postprocess(self, image_list, feats):
        self._features = {"image_embed": feats[0], "high_res_feats": feats[1]}
        self._is_image_set = True

        if len(image_list) == 1:
            self._is_batch = False
        else:
            self._is_batch = True


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    ax.axis("off")


def initialize(predictor, batch_size):
    module = torch.compile(
        predictor.set_image,
        backend="rbln",
        options={"cache_dir": "./.cache"},
        dynamic=False,
    )

    return module, predictor


def save_segmented_image(
    image_batch,
    pts_batch,
    masks_batch,
    labels_batch,
    scores_batch,
):
    if len(image_batch) == 1:
        masks_batch = [masks_batch]
        scores_batch = [scores_batch]

    # Select the best single mask per object
    best_masks = []
    for masks, scores in zip(masks_batch, scores_batch):
        best_masks.append(masks[range(len(masks)), np.argmax(scores, axis=-1)])

    for i, (image, points, best_mask, labels, score) in enumerate(
        zip(image_batch, pts_batch, best_masks, labels_batch, scores_batch)
    ):
        plt.figure(figsize=(10, 10))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray_image, cmap="gray")

        for mask in best_mask:
            show_mask(mask, plt.gca(), random_color=True)

        plt.savefig(
            f"segmented_object_{i}.jpg",
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close()


def main():
    # Set up arguments
    args = parsing_argument()
    cfg = CFG_CONFIG[args.model_name]

    if not os.path.isfile(cfg["checkpoint"]):
        print("Checkpoint not found. Downloading model checkpoint script...")
        run_download_script()

    images_dir = "images"
    if not os.path.isdir(images_dir):
        print("Input image not found. Downloading input image...")

        os.makedirs(images_dir, exist_ok=True)
        print(f"Directory '{images_dir}' created or already exists.")

        download_file(
            "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg",
            os.path.join(images_dir, "groceries.jpg"),
        )
        download_file(
            "https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/notebooks/images/truck.jpg",
            os.path.join(images_dir, "trunk.jpg"),
        )

    # Initialize model
    model = build_sam2(cfg["config_file"], cfg["checkpoint"], device="cpu")
    predictor = WrapperSAM2ImagePredictor(model)
    module, predictor = initialize(predictor, args.batch_size)

    # Load images and points
    image_batch = [np.array(Image.open("images/groceries.jpg").convert("RGB"))]
    pts_batch = [np.array([[[400, 300]], [[630, 300]]])]
    labels_batch = [np.array([[1], [1]])]

    # Load second image and points based on args.batch_size
    for _ in range(args.batch_size - 1):
        image_batch.append(np.array(Image.open("images/trunk.jpg").convert("RGB")))
        pts_batch.append(np.array([[[500, 375]], [[650, 750]]]))
        labels_batch.append(np.array([[1], [1]]))

    # Run phase
    feats = module(predictor.set_image_preprocess(image_batch))
    predictor.set_image_postprocess(image_batch, feats)

    if len(image_batch) == 1:
        masks_batch, scores_batch, logits_batch = predictor.predict(
            pts_batch[0], labels_batch[0]
        )
    else:
        masks_batch, scores_batch, logits_batch = predictor.predict_batch(
            pts_batch, labels_batch, box_batch=None, multimask_output=True
        )

    # Save segmented images
    save_segmented_image(
        image_batch, pts_batch, masks_batch, labels_batch, scores_batch
    )


if __name__ == "__main__":
    main()
