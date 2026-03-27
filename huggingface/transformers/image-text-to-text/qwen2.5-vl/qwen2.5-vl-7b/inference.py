import atexit
import os
import tempfile
import urllib.parse
import urllib.request
from contextlib import suppress

from optimum.rbln import RBLNAutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


def _cleanup_tmpfile(path: str) -> None:
    with suppress(OSError):
        os.unlink(path)


def _video_url_to_local_path(url: str) -> str:
    """Download URL to a temp file for video decoding."""
    if not url.strip().lower().startswith(("http://", "https://")):
        return url
    suffix = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    atexit.register(_cleanup_tmpfile, path)
    urllib.request.urlretrieve(url, path)
    return path


def main():
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_dir = os.path.basename(model_id)

    # Load compiled model
    processor = AutoProcessor.from_pretrained(model_dir)
    model = RBLNAutoModelForVision2Seq.from_pretrained(
        model_dir,
        export=False,
        rbln_config={
            "visual": {
                # The `device` parameter specifies which device should be used for each submodule during runtime.
                "device": 0,
            },
            "device": [0, 1, 2, 3, 4, 5, 6, 7],
        },
    )

    # Messages containing a video url and a text query
    video_url = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4"
    )
    video_path = _video_url_to_local_path(video_url)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    # In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        # Minimum and maximum pixel constraints for image or video processing, defined as patch counts.
        # Example: Set min_pixels and max_pixels to a patch range of 1024 to 5120 to balance performance and computational cost.
        # The max_pixels setting is closely tied to the visual model's max_seq_lens, as it determines the maximum number of patches processed.
        min_pixels=1024
        * 14
        * 14,  # Minimum resolution in pixels (e.g., 1024 patches at patch size 14).
        max_pixels=5120
        * 14
        * 14,  # Maximum resolution in pixels (e.g., 5120 patches at patch size 14).
        **video_kwargs,
    )

    # autoregressively complete prompt
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    input_len = inputs.input_ids.shape[-1]
    generated_ids_trimmed = generated_ids[0][input_len:]

    # Show text and result
    print("--Result--")
    print(processor.decode(generated_ids_trimmed, skip_special_tokens=True))


if __name__ == "__main__":
    main()
