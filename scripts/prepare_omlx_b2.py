#!/usr/bin/env python3
"""oMLX views of the bench-2 MTPLX packs (same weights as MTPLX and Splash) and their oMLX base paths.

For each pack, models/<id>/ is made of symlinks to the pack's files, except:
  - mtp.safetensors becomes model-mtp.safetensors with oMLX's names (language_model.mtp.*, the
    layout of Jundot's -mtp checkpoints); a MoE MTP head's numbered experts are stacked into
    switch_mlp (mlx-lm's layout). Packed codes, scales and biases are copied bit for bit.
  - config.json gets a quantization entry for every quantized MTP module (bits and group size
    read from the tensor shapes); the index lists the MTP tensors.
engines/omlx-b2-<id>/ gets settings.json + model_settings.json from templates/omlx/*-q8.json
(TurboQuant 8-bit KV, MTP on, 3 draft tokens, thinking on), so ~/.omlx is never touched.

  python scripts/prepare_omlx_b2.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
LOCAL = json.loads((ROOT / "config" / "local.json").read_text())
HUB = Path.home() / ".cache" / "huggingface" / "hub"
PACKS = {  # name: (source, MTP head in an MTPLX sidecar)
    "b2-qwen38-27b-mlxc": (Path(LOCAL["B2_MODEL_27B"]), True),
    "b2-qwen36-a3b-mlxc": (Path(LOCAL["B2_MODEL_A3B"]), True),
    "jundot-qwen3-8-27b-oq4e-fp16-mtp": (next((HUB / "models--Jundot--Qwen3.8-27B-oQ4e-fp16-mtp" / "snapshots").iterdir()), False),
    "jundot-qwen3-6-35b-a3b-oq4e-mtp": (next((HUB / "models--Jundot--Qwen3.6-35B-A3B-oQ4e-mtp" / "snapshots").iterdir()), False),
}
VISION_FILES = {"preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
                "model-vision.safetensors"}
SKIP = {"config.json", "model.safetensors.index.json", "mtp.safetensors", "mtplx_runtime.json",
        "MTPLX_FP16_CONVERSION_MANIFEST.json", "MTPLX_PUBLISH_MANIFEST.json"}
EXPERT = re.compile(r"^mtp\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$")


def mtp_tensors(src: Path) -> dict[str, mx.array]:
    raw = mx.load(str(src / "mtp.safetensors"))
    out: dict[str, mx.array] = {}
    stacks: dict[tuple, dict[int, mx.array]] = {}
    for k, v in raw.items():
        m = EXPERT.match(k)
        if m:
            layer, e, proj, part = m.groups()
            stacks.setdefault((layer, proj, part), {})[int(e)] = v
        else:
            out["language_model." + k] = v
    for (layer, proj, part), by_e in stacks.items():
        out[f"language_model.mtp.layers.{layer}.mlp.switch_mlp.{proj}.{part}"] = mx.stack(
            [by_e[i] for i in range(len(by_e))])
    return out


def quant_entries(tensors: dict[str, mx.array], recorded: dict) -> dict[str, dict]:
    """One entry per quantized MTP module, at the bits and group size the pack records
    (the shapes alone cannot tell 4-bit g64 from 2-bit g128)."""
    bits, group = int(recorded["bits"]), int(recorded["group_size"])
    q = {}
    for k in tensors:
        if k.endswith(".scales"):
            mod = k[: -len(".scales")]
            w, s = tensors[mod + ".weight"], tensors[k]
            assert w.shape[-1] * 32 // bits == s.shape[-1] * group, (mod, w.shape, s.shape)
            q[mod] = {"bits": bits, "group_size": group, "mode": "affine"}
    return q


def view(name: str, src: Path, sidecar: bool) -> None:
    """Text-only view: oMLX 0.7.1 serves a checkpoint with a vision_config through mlx-vlm, whose
    MTP path cannot run with TurboQuant KV; without it (and without the vision files) the
    checkpoint goes through mlx-lm, with MTP and 8-bit KV together."""
    dst = ROOT / "models" / name
    if dst.is_symlink():
        dst.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name not in SKIP and f.name not in VISION_FILES and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f.resolve())
    index = json.loads((src / "model.safetensors.index.json").read_text())
    index["weight_map"] = {k: v for k, v in index["weight_map"].items() if not k.startswith("vision_tower.")}
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("vision_config", None)
    entries = {}
    if sidecar:
        tensors = mtp_tensors(src)
        mx.save_safetensors(str(dst / "model-mtp.safetensors"), tensors, metadata={"format": "mlx"})
        index["weight_map"].update({k: "model-mtp.safetensors" for k in tensors})
        entries = quant_entries(tensors, cfg["mtplx_mtp_quantization"])
        for key in ("mlx_lm_extra_tensors", "mtplx_mtp_expected_keys"):
            cfg.pop(key, None)
        cfg["quantization"].update(entries)
        if "quantization_config" in cfg:
            cfg["quantization_config"] = cfg["quantization"]
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    print(name, "MTP entries", sorted({json.dumps(v) for v in entries.values()}))


def base_path(name: str) -> None:
    base = ROOT / "engines" / f"omlx-b2-{name}"
    base.mkdir(parents=True, exist_ok=True)
    settings = json.loads((ROOT / "templates" / "omlx" / "settings-q8.json").read_text()
                          .replace("${ROOT}", str(ROOT)).replace("omlx-base-q8", f"omlx-b2-{name}"))
    (base / "settings.json").write_text(json.dumps(settings, indent=2))
    ms = json.loads((ROOT / "templates" / "omlx" / "model_settings-q8.json").read_text())
    entry = next(iter(ms["models"].values()))
    entry.update(enable_thinking=True, mtp_enabled=True, mtp_num_draft_tokens=3)
    ms["models"] = {name: entry}
    (base / "model_settings.json").write_text(json.dumps(ms, indent=2))


if __name__ == "__main__":
    for name, (src, sidecar) in PACKS.items():
        view(name, src, sidecar)
        if not (ROOT / "engines" / f"omlx-b2-{name}" / "model_settings.json").exists():
            base_path(name)
