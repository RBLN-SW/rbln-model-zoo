import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TripoSR"))

import imageio.v2 as imageio
import numpy as np
import rembg
from modeling_triposr import RBLNTripoSR
from PIL import Image, ImageDraw
from tsr.utils import remove_background, resize_foreground


def render_rotating_gif(mesh, output_path, n_frames=36, size=320):
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vcolors = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0

    verts = verts - (verts.min(0) + verts.max(0)) / 2.0
    verts = verts / (np.linalg.norm(verts, axis=1).max() + 1e-8)

    face_colors = vcolors[faces].mean(axis=1)
    scale = size * 0.45

    frames = []
    for i in range(n_frames):
        theta = 2.0 * np.pi * i / n_frames
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        v = verts @ R.T

        x = scale * v[:, 0] + size / 2
        y = -scale * v[:, 1] + size / 2

        n = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12

        shade = 0.3 + 0.7 * np.clip(n[:, 2], 0, 1)
        rgb = (face_colors * shade[:, None] * 255).astype(np.uint8)

        vis = n[:, 2] > 0
        order = np.where(vis)[0][np.argsort(v[faces[vis], 2].mean(1))]

        img = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(img)
        for fi in order:
            tri = faces[fi]
            draw.polygon(list(zip(x[tri], y[tri])), fill=tuple(rgb[fi].tolist()))
        frames.append(np.asarray(img))

    imageio.mimsave(output_path, frames, duration=0.05, loop=0)


def main():
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    model = RBLNTripoSR.from_pretrained(
        os.path.abspath("./triposr"),
        export=False,
    )

    image_path = "./TripoSR/examples/chair.png"

    rembg_session = rembg.new_session()

    image = remove_background(Image.open(image_path), rembg_session)
    image = resize_foreground(image, 0.85)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    image.save(os.path.join(output_dir, "input.png"))

    scene_codes = model([image])

    meshes = model.extract_mesh(scene_codes, True, resolution=256)

    out_mesh_path = os.path.join(output_dir, "mesh.obj")
    meshes[0].export(out_mesh_path)

    out_gif_path = os.path.join(output_dir, "mesh.gif")
    render_rotating_gif(meshes[0], out_gif_path)


if __name__ == "__main__":
    main()
