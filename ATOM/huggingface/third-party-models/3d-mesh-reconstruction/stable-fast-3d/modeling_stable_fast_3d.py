import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stable-fast-3d"))

import rebel
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sf3d.system import SF3D

_HEAVY_MODULES = ("image_tokenizer", "backbone", "post_processor")
_IMAGE_ESTIMATOR = "image_estimator"


class _RuntimeModule(nn.Module):
    def __init__(self, runtime: "rebel.Runtime"):
        super().__init__()
        self.runtime = runtime

    def forward(self, *args, **kwargs):
        return self.runtime(*args, *(v for v in kwargs.values() if v is not None))


class _ClipBodyCompileWrapper(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, cond_image: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(cond_image)


class _ClipModelStub(nn.Module):
    def __init__(self, runtime: "rebel.Runtime"):
        super().__init__()
        self.runtime = runtime

    def encode_image(self, cond_image: torch.Tensor) -> torch.Tensor:
        return self.runtime(cond_image)


def _input_specs(model: SF3D, batch_size: int) -> Dict[str, list]:
    cond = model.cfg.cond_image_size
    n_image_tokens = (cond // 14) ** 2 + 1
    hidden = model.image_tokenizer.model.config.hidden_size
    plane = model.tokenizer.cfg.plane_size
    chans = model.tokenizer.cfg.num_channels
    n_triplane = 3 * plane * plane
    pp_in = model.post_processor.cfg.in_channels
    cam_dim = model.image_tokenizer.cfg.modulation_cond_dim
    return {
        "image_tokenizer": [
            ("images", [batch_size, 1, 3, cond, cond], "float32"),
            ("modulation_cond", [batch_size, 1, cam_dim], "float32"),
        ],
        "backbone": [
            ("hidden_states", [batch_size, chans, n_triplane], "float32"),
            ("encoder_hidden_states", [batch_size, n_image_tokens, hidden], "float32"),
        ],
        "post_processor": [
            ("triplanes", [batch_size, 3, pp_in, plane, plane], "float32"),
        ],
        _IMAGE_ESTIMATOR: [
            ("cond_image", [batch_size, 3, 224, 224], "float32"),
        ],
    }


def _swap_runtimes(sf3d: SF3D, runtimes: Dict[str, "rebel.Runtime"]) -> None:
    for name, rt in runtimes.items():
        if name == _IMAGE_ESTIMATOR:
            sf3d.image_estimator.model = _ClipModelStub(rt)
        else:
            setattr(sf3d, name, _RuntimeModule(rt))


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


class RBLNSF3D:
    def __init__(
        self,
        sf3d: SF3D,
        model_save_dir: Union[str, Path, TemporaryDirectory],
    ):
        self._sf3d = sf3d
        if isinstance(model_save_dir, TemporaryDirectory):
            self._tmpdir = model_save_dir
            self.model_save_dir = Path(model_save_dir.name)
        else:
            self._tmpdir = None
            self.model_save_dir = Path(model_save_dir)

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        export: bool,
        rbln_batch_size: int = 1,
        rbln_image_size: int = 512,
        rbln_device: Union[int, list] = 0,
    ) -> "RBLNSF3D":
        if export:
            return cls._compile(
                str(model_id), rbln_batch_size, rbln_image_size, rbln_device
            )
        return cls._load(str(model_id), rbln_device)

    @classmethod
    def _compile(
        cls,
        model_id: str,
        batch_size: int,
        image_size: int,
        device,
    ) -> "RBLNSF3D":
        save_dir = TemporaryDirectory()
        save_path = Path(save_dir.name)

        sf3d = SF3D.from_pretrained(
            model_id, config_name="config.yaml", weight_name="model.safetensors"
        )
        sf3d.eval()
        specs = _input_specs(sf3d, batch_size)

        modules: Dict[str, nn.Module] = {n: getattr(sf3d, n) for n in _HEAVY_MODULES}
        modules[_IMAGE_ESTIMATOR] = _ClipBodyCompileWrapper(sf3d.image_estimator.model)

        for name, module in modules.items():
            cm = rebel.compile_from_torch(module, input_info=specs[name])
            cm.save(str(save_path / f"{name}.rbln"))

        OmegaConf.save(sf3d.cfg, save_path / "config.yaml")
        torch.save(
            {
                "tokenizer": sf3d.tokenizer.state_dict(),
                "camera_embedder": sf3d.camera_embedder.state_dict(),
                "decoder": sf3d.decoder.state_dict(),
                "image_estimator_heads": sf3d.image_estimator.heads.state_dict(),
            },
            save_path / "lightweight.pth",
        )

        return cls(sf3d, save_dir)

    @classmethod
    def _load(cls, model_id: str, device) -> "RBLNSF3D":
        path = Path(model_id)
        cfg = OmegaConf.load(path / "config.yaml")
        light = torch.load(
            path / "lightweight.pth", weights_only=False, map_location="cpu"
        )

        sf3d = SF3D(cfg)
        sf3d.eval()
        sf3d.tokenizer.load_state_dict(light["tokenizer"])
        sf3d.camera_embedder.load_state_dict(light["camera_embedder"])
        sf3d.decoder.load_state_dict(light["decoder"])
        sf3d.image_estimator.heads.load_state_dict(light["image_estimator_heads"])

        runtimes: Dict[str, "rebel.Runtime"] = {
            f.stem: rebel.Runtime(
                rebel.RBLNCompiledModel(f), tensor_type="pt", device=device
            )
            for f in path.glob("*.rbln")
        }
        _swap_runtimes(sf3d, runtimes)
        return cls(sf3d, path)

    def run_image(self, *args, **kwargs):
        return self._sf3d.run_image(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._sf3d(*args, **kwargs)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        _copy_tree(self.model_save_dir, Path(save_directory))
