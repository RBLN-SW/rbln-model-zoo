# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import imageio
import numpy as np
import torch
from optimum.rbln import RBLNDepthAnythingForDepthEstimation
from PIL import Image
from transformers import AutoImageProcessor


class DepthAnythingModel:
    def __init__(
        self,
        export: bool,
        model_save_dir: Optional[Union[str, Path]],
        device: Dict[str, Any] = None,
    ):
        """
        Initialize the Depth Anything model and its image processor.
        """
        self.device = torch.device("cpu")
        self.model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_id,
        )

        if export:
            self.model = RBLNDepthAnythingForDepthEstimation.from_pretrained(
                self.model_id,
                export=export,
                rbln_image_size=(518, 518),
                rbln_optimize_host_memory=False,
                rbln_create_runtimes=False,
            )
            self.model.save_pretrained(os.path.join(model_save_dir, "depth_anything"))
        else:
            self.model = RBLNDepthAnythingForDepthEstimation.from_pretrained(
                model_id=model_save_dir + "/depth_anything",
                export=export,
                rbln_device=device["depth_anything"],
            )

    def predict_depth(self, image: Image.Image) -> Image.Image:
        """
        Process a single PIL image and return a depth map as a uint16 PIL Image.
        """

        # pad image for rbln
        width, height = image.size
        max_side = max(width, height)
        square_image = Image.new("RGB", (max_side, max_side), (0, 0, 0))
        square_image.paste(image, (0, 0))

        # Prepare inputs for the model
        inputs = self.image_processor(images=square_image, return_tensors="pt")

        # Move all tensors to the proper device with half precision
        inputs = {k: v.to(self.device, dtype=torch.float32) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate the predicted depth to the original image size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=max_side,
            mode="bicubic",
            align_corners=False,
        )

        unpadded_image = prediction[:, :height, :width]

        # Convert the output tensor to a numpy array and save as a depth image
        output = unpadded_image.squeeze().cpu().numpy()
        depth_image = DepthAnythingModel.save_depth(output)
        return depth_image

    def __call__(self, input_video: str, output_video: str = "depth.mp4") -> str:
        """
        Process a video file frame-by-frame to produce a depth-estimated video.
        The output video is saved as an MP4 file.
        """

        assert os.path.exists(input_video)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return

        # Retrieve video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        depths = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame from BGR to RGB and then to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            width, height = image.size
            max_side = max(width, height)
            square_image = Image.new("RGB", (max_side, max_side), (0, 0, 0))
            square_image.paste(image, (0, 0))
            inputs = self.image_processor(images=square_image, return_tensors="pt")
            inputs = {
                k: v.to(self.device, dtype=torch.float32) for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # For video processing, take the first output and interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth[0].unsqueeze(0).unsqueeze(0),
                # size=(frame_height, frame_width),
                size=max_side,
                mode="bicubic",
                align_corners=False,
            )

            unpadded_image = prediction[:, :, :frame_height, :frame_width]
            depth = unpadded_image.squeeze().cpu().numpy()
            depths += [depth]
        cap.release()

        depths = np.stack(depths)
        depths_normed = (
            (depths - depths.min()) / (depths.max() - depths.min() + 1e-8) * 255.0
        )
        depths_normed = depths_normed.astype(np.uint8)

        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        self.write_video(depths_normed, output_video, fps=fps)
        return output_video

    @staticmethod
    def save_depth(output: np.ndarray) -> Image.Image:
        """
        Convert the raw depth output (float values) into a uint16 PIL Image.
        """
        depth_min = output.min()
        depth_max = output.max()
        max_val = (2**16) - 1  # Maximum value for uint16

        if depth_max - depth_min > np.finfo("float").eps:
            out_array = max_val * (output - depth_min) / (depth_max - depth_min)
        else:
            out_array = np.zeros_like(output)

        formatted = out_array.astype("uint16")
        depth_image = Image.fromarray(formatted, mode="I;16")
        return depth_image

    @staticmethod
    def write_video(frames, output_path, fps=30):
        with imageio.get_writer(output_path, fps=fps, macro_block_size=8) as writer:
            for frame in frames:
                if len(frame.shape) == 2:  # single channel
                    frame = frame[:, :, None].repeat(3, axis=2)
                writer.append_data(frame)
