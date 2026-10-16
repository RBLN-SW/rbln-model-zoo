import urllib

import numpy as np
import rebel  # RBLN Runtime
import torch  # for Preprocessing
import torch.nn.functional as F
from PIL import Image


def preprocess(mask_values, pil_img, scale, is_mask):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)

    assert newW > 0 and newH > 0, (
        "Scale is too small, resized images would have no pixel"
    )
    pil_img = pil_img.resize(
        (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC
    )
    img = np.asarray(pil_img)

    if is_mask:
        mask = np.zeros((newH, newW), dtype=np.int64)
        for i, v in enumerate(mask_values):
            if img.ndim == 2:
                mask[img == v] = i
            else:
                mask[(img == v).all(-1)] = i

        return mask

    else:
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img


def postprocess(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros(
            (mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8
        )
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def main():
    # Download and example image
    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/car.jpg"
    img_path = "./car.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())

    input_img = Image.open(img_path)
    # The size of the input image will be (480, 320) after passing the preprocess() function.
    input_img = input_img.resize((960, 640))
    img = torch.from_numpy(preprocess(None, input_img, 0.5, is_mask=False))
    img = img.unsqueeze(0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    module = rebel.Runtime("./unet.rbln")
    output = module.run(img)

    # postprocess
    rebel_out = F.interpolate(
        torch.tensor(output), (input_img.size[1], input_img.size[0]), mode="bilinear"
    )
    mask = rebel_out.argmax(dim=1)
    mask = mask[0].long().squeeze().numpy()
    result = postprocess(mask, [0, 1])
    result.save("unet_result.png")


if __name__ == "__main__":
    main()
