import urllib.request

import numpy as np
import rebel
import tensorflow as tf
from PIL import Image


def preprocess(input_image, input_size):
    image = tf.io.read_file(input_image)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=[input_size, input_size])
    image = image / 127.5 - 1
    return image


def postprocess(mask, n_classes):
    colormap = np.array(
        [
            [0, 0, 0],
            [31, 119, 180],
            [44, 160, 44],
            [44, 127, 125],
            [52, 225, 143],
            [217, 222, 163],
            [254, 128, 37],
            [130, 162, 128],
            [121, 7, 166],
            [136, 183, 248],
            [85, 1, 76],
            [22, 23, 62],
            [159, 50, 15],
            [101, 93, 152],
            [252, 229, 92],
            [167, 173, 17],
            [218, 252, 252],
            [238, 126, 197],
            [116, 157, 140],
            [214, 220, 252],
        ],
        dtype=np.uint8,
    )

    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for i in range(0, n_classes):
        idx = mask == i
        r[idx] = colormap[i, 0]
        g[idx] = colormap[i, 1]
        b[idx] = colormap[i, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def main():
    input_size = 512

    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/people.jpg"
    img_path = "./people.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())

    input_batch = preprocess(img_path, input_size)
    batch = np.expand_dims((input_batch), axis=0)

    module = rebel.Runtime("deeplabv3.rbln")

    rebel_result = module.run(batch)
    rebel_result = np.squeeze(rebel_result)
    rebel_result = np.argmax(rebel_result, axis=2)

    rebel_out_image = postprocess(rebel_result, 20)
    rebel_out_image = Image.fromarray(rebel_out_image)
    rebel_out_image.save("rebel.jpg")


if __name__ == "__main__":
    main()
