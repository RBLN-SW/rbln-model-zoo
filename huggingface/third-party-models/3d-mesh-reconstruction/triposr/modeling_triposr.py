import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TripoSR"))

import rebel
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tsr.system import TSR

_HEAVY_MODULES = ("image_tokenizer", "backbone", "post_processor")
_DECODER = "decoder"


class _RuntimeModule(nn.Module):
    def __init__(self, runtime: "rebel.Runtime"):
        super().__init__()
        self.runtime = runtime

    def forward(self, *args, **kwargs):
        return self.runtime(*args, *(v for v in kwargs.values() if v is not None))


class _NeRFMLPCompileWrapper(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(x)
        return torch.cat([out["density"], out["features"]], dim=-1)


class _RBLNNeRFMLP(nn.Module):
    def __init__(self, runtime: "rebel.Runtime", chunk_size: int):
        super().__init__()
        self.runtime = runtime
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outs = []
        for i in range(0, x.shape[0], self.chunk_size):
            c = x[i : i + self.chunk_size]
            if c.shape[0] == self.chunk_size:
                outs.append(self.runtime(c))
            else:
                pad = torch.zeros(
                    self.chunk_size, c.shape[-1], dtype=c.dtype, device=c.device
                )
                pad[: c.shape[0]] = c
                outs.append(self.runtime(pad)[: c.shape[0]])
        out = torch.cat(outs, 0)
        return {"density": out[..., 0:1], "features": out[..., 1:4]}


def _input_specs(model: TSR, batch_size: int, chunk_size: int):
    cond = model.cfg.cond_image_size
    n_image_tokens = (cond // 16) ** 2 + 1
    hidden = model.image_tokenizer.model.config.hidden_size
    plane = model.tokenizer.cfg.plane_size
    chans = model.tokenizer.cfg.num_channels
    n_triplane = 3 * plane * plane
    pp_in = model.post_processor.cfg.in_channels
    return {
        "image_tokenizer": [("images", [batch_size, 1, 3, cond, cond], "float32")],
        "backbone": [
            ("hidden_states", [batch_size, chans, n_triplane], "float32"),
            ("encoder_hidden_states", [batch_size, n_image_tokens, hidden], "float32"),
        ],
        "post_processor": [
            ("triplanes", [batch_size, 3, pp_in, plane, plane], "float32"),
        ],
        _DECODER: [
            ("x", [chunk_size, model.decoder.cfg.in_channels], "float32"),
        ],
    }


def _swap_runtimes(
    tsr: TSR, runtimes: Dict[str, "rebel.Runtime"], chunk_size: int
) -> None:
    for name, rt in runtimes.items():
        if name == _DECODER:
            tsr.decoder = _RBLNNeRFMLP(rt, chunk_size)
        else:
            setattr(tsr, name, _RuntimeModule(rt))


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


class RBLNTripoSR:
    def __init__(
        self,
        tsr: TSR,
        model_save_dir: Union[str, Path, TemporaryDirectory],
        chunk_size: int,
    ):
        self._tsr = tsr
        self.chunk_size = chunk_size
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
        rbln_chunk_size: int = 8192,
        rbln_device: Union[int, list] = 0,
    ) -> "RBLNTripoSR":
        if export:
            return cls._compile(
                str(model_id),
                rbln_batch_size,
                rbln_image_size,
                rbln_chunk_size,
                rbln_device,
            )
        return cls._load(str(model_id), rbln_device)

    @classmethod
    def _compile(
        cls,
        model_id: str,
        batch_size: int,
        image_size: int,
        chunk_size: int,
        device,
    ) -> "RBLNTripoSR":
        save_dir = TemporaryDirectory()
        save_path = Path(save_dir.name)

        tsr = TSR.from_pretrained(model_id, "config.yaml", "model.ckpt")
        tsr.eval()
        specs = _input_specs(tsr, batch_size, chunk_size)

        modules: Dict[str, nn.Module] = {n: getattr(tsr, n) for n in _HEAVY_MODULES}
        modules[_DECODER] = _NeRFMLPCompileWrapper(tsr.decoder)

        for name, module in modules.items():
            cm = rebel.compile_from_torch(module, input_info=specs[name])
            cm.save(str(save_path / f"{name}.rbln"))

        OmegaConf.save(tsr.cfg, save_path / "config.yaml")
        torch.save(
            {
                "chunk_size": chunk_size,
                "tokenizer": tsr.tokenizer.state_dict(),
                "renderer": tsr.renderer.state_dict(),
            },
            save_path / "lightweight.pth",
        )

        return cls(tsr, save_dir, chunk_size)

    @classmethod
    def _load(cls, model_id: str, device) -> "RBLNTripoSR":
        path = Path(model_id)
        cfg = OmegaConf.load(path / "config.yaml")
        light = torch.load(
            path / "lightweight.pth", weights_only=False, map_location="cpu"
        )

        tsr = TSR(cfg)
        tsr.eval()
        tsr.tokenizer.load_state_dict(light["tokenizer"])
        tsr.renderer.load_state_dict(light["renderer"])

        runtimes: Dict[str, "rebel.Runtime"] = {
            f.stem: rebel.Runtime(
                rebel.RBLNCompiledModel(f), tensor_type="pt", device=device
            )
            for f in path.glob("*.rbln")
        }
        _swap_runtimes(tsr, runtimes, light["chunk_size"])
        return cls(tsr, path, light["chunk_size"])

    def __call__(self, image, **kwargs) -> torch.Tensor:
        return self._tsr(image, "cpu")

    def extract_mesh(
        self,
        scene_codes: torch.Tensor,
        has_vertex_color: bool = True,
        resolution: int = 256,
        threshold: float = 25.0,
    ):
        return self._tsr.extract_mesh(
            scene_codes,
            has_vertex_color,
            resolution=resolution,
            threshold=threshold,
        )

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        _copy_tree(self.model_save_dir, Path(save_directory))
