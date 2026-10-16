import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import rebel
import torch


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "yolo11n-seg",
            "yolo11s-seg",
            "yolo11m-seg",
            "yolo11l-seg",
            "yolo11x-seg",
        ],
        default="yolo11n-seg",
        help="YOLO segmentation model name",
    )
    return parser.parse_args()


def download_image(url: str, save_path: Path) -> np.ndarray:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp, open(save_path, "wb") as f:
        f.write(resp.read())
    return cv2.imread(str(save_path))


def preprocess(image: np.ndarray) -> np.ndarray:
    from ultralytics.data.augment import LetterBox

    img = LetterBox(new_shape=(640, 640))(image=image)
    img = img.transpose((2, 0, 1))[::-1]
    return (img[None] / 255).astype(np.float32).copy()


def load_model_config(pt_path: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(pt_path)
    num_classes = len(model.names)
    box_cls_dim = num_classes + 4
    return dict(
        num_classes=num_classes,
        box_cls_dim=box_cls_dim,
        conf=model.overrides.get("conf", 0.25),
        iou=model.overrides.get("iou", 0.45),
        max_det=model.overrides.get("max_det", 1000),
        names=model.names,
        agnostic_nms=model.overrides.get("agnostic_nms", False),
        classes=model.overrides.get("classes", None),
        retina_masks=False,
    )


def postprocess(preds, batch, orig_imgs, img_path: str, cfg: dict):
    from ultralytics.engine.results import Results
    from ultralytics.utils import ops

    box_cls = torch.from_numpy(preds[0][:, : cfg["box_cls_dim"], :])
    masks = torch.from_numpy(preds[4])
    new_preds = [
        torch.cat((box_cls, masks), dim=1),
        tuple(torch.from_numpy(p) for p in preds[1:6]),
    ]
    p = ops.non_max_suppression(
        new_preds[0],
        cfg["conf"],
        cfg["iou"],
        agnostic=cfg["agnostic_nms"],
        max_det=cfg["max_det"],
        nc=cfg["num_classes"],
        classes=cfg["classes"],
    )
    if not isinstance(orig_imgs, list):
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    proto = new_preds[1][-1] if isinstance(new_preds[1], tuple) else new_preds[1]
    results = []
    for pred, orig_img, path in zip(p, orig_imgs, [img_path]):
        if not len(pred):
            masks = None
        elif cfg["retina_masks"]:
            pred[:, :4] = ops.scale_boxes(batch.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(
                proto[0], pred[:, 6:], pred[:, :4], orig_img.shape[:2]
            )
        else:
            masks = ops.process_mask(
                proto[0], pred[:, 6:], pred[:, :4], batch.shape[2:], upsample=True
            )
            pred[:, :4] = ops.scale_boxes(batch.shape[2:], pred[:, :4], orig_img.shape)
        results.append(
            Results(
                orig_img, path=path, names=cfg["names"], boxes=pred[:, :6], masks=masks
            )
        )
    return results


def main():
    args = parsing_argument()
    pt_path = f"{args.model_name}.pt"

    # Set config
    cfg = load_model_config(pt_path)

    # Load compiled models
    module = rebel.Runtime(f"{args.model_name}.rbln")

    # Load input images
    img_path = Path("people.jpg")
    img = download_image("https://ultralytics.com/images/bus.jpg", img_path)

    # Foward
    batch = preprocess(img)
    preds = module.run(batch)
    results = postprocess(preds, torch.from_numpy(batch), [img], str(img_path), cfg)

    # Save images
    for i, r in enumerate(results):
        r.save(filename=f"results{i}.jpg")


if __name__ == "__main__":
    main()
