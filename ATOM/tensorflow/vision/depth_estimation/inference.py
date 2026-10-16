import urllib.request

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import rebel
import tensorflow as tf

tf.random.set_seed(123)


def preprocess(image):
    input_image = []
    x = np.clip(image / 255, 0, 1)
    input_image.append(x)
    return np.stack(input_image, axis=0)


def depth_norm(x, maxDepth):
    return maxDepth / x


def postprocess(outputs, plasma, minDepth=10, maxDepth=1000):
    x = np.clip(depth_norm(outputs, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth
    rescaled = x[0][:, :, 0]
    rescaled = rescaled - np.min(rescaled)
    rescaled = rescaled / np.max(rescaled)
    out_image = plasma(rescaled)[:, :, :3]
    return out_image


def main():
    input_h = 480
    input_w = 640

    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/classroom.jpg"
    img_path = "./classroom.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())

    img = cv2.imread("./classroom.jpg")
    resized_img = cv2.resize(img, (input_w, input_h)).astype(np.float32)
    input_batch = preprocess(resized_img)

    if len(input_batch.shape) < 3:
        input_batch = np.stack((input_batch, input_batch, input_batch), axis=2)
    if len(input_batch.shape) < 4:
        input_batch = input_batch.reshape(
            (1, input_batch.shape[0], input_batch.shape[1], input_batch[2])
        )

    plasma = plt.get_cmap("plasma")

    module = rebel.Runtime("monocular_depth_estimation.rbln")
    rebel_result = module.run(input_batch)
    save_rebel_img = postprocess(rebel_result, plasma=plasma)
    imageio.imsave("rebel.jpg", (save_rebel_img * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
