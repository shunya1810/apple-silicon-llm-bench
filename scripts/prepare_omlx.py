#!/usr/bin/env python3
"""Prepare oMLX to serve the same MTPLX model files, plus isolated oMLX base paths.

1. models/qwen38-27b-mtplx-speed/: a text-only view of ${MTPLX_MODEL} made of symlinks.
   - the vision tower (model-vision.safetensors, *processor_config.json) is left out, so
     oMLX loads the checkpoint as a text LLM (its MTP path is on the mlx-lm side);
   - mtp.safetensors is exposed as model-mtp.safetensors (mlx-lm loads model*.safetensors)
     and its tensors are added to the index;
   - config.json gets per-module quantization entries for the MTP head (4-bit, group 64,
     as recorded in mtplx_runtime.json); the body default in the file is group 32.
   No weight is converted or copied.
2. engines/omlx-base-{q8,fp16}/: settings.json + model_settings.json for `omlx serve
   --base-path` from templates/omlx/, so the benchmark never touches ~/.omlx. q8 =
   TurboQuant 8-bit KV, fp16 = KV unquantized; both with Lightning MTP, 3 draft tokens,
   thinking off. oMLX writes its own auth secret into settings.json on first start.

  python scripts/prepare_omlx.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = json.loads((ROOT / "config" / "local.json").read_text())
SRC = Path(LOCAL["MTPLX_MODEL"])
DST = ROOT / "models" / "qwen38-27b-mtplx-speed"
MODEL_ID = DST.name
SKIP = {"model-vision.safetensors", "preprocessor_config.json", "processor_config.json",
        "video_preprocessor_config.json", "model.safetensors.index.json", "mtp.safetensors", "config.json"}


def safetensors_keys(path: Path) -> list[str]:
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return [k for k in header if k != "__metadata__"]


def model_view() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for f in SRC.iterdir():
        if f.name not in SKIP and not (DST / f.name).exists():
            (DST / f.name).symlink_to(f)
    if not (DST / "model-mtp.safetensors").exists():
        (DST / "model-mtp.safetensors").symlink_to(SRC / "mtp.safetensors")
    mtp_keys = safetensors_keys(SRC / "mtp.safetensors")

    index = json.loads((SRC / "model.safetensors.index.json").read_text())
    wm = {k: v for k, v in index["weight_map"].items() if v != "model-vision.safetensors"}
    wm.update({k: "model-mtp.safetensors" for k in mtp_keys})
    index["weight_map"] = wm
    (DST / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))

    runtime = json.loads((SRC / "mtplx_runtime.json").read_text())
    bits = int(runtime.get("mtp_quant_bits", 4))
    group = int(runtime.get("mtp_quant_group_size", 64))
    cfg = json.loads((SRC / "config.json").read_text())
    q = cfg["quantization"]
    for k in mtp_keys:
        if k.endswith(".scales"):
            q["language_model." + k[: -len(".scales")]] = {"bits": bits, "group_size": group, "mode": "affine"}
    if "quantization_config" in cfg:
        cfg["quantization_config"] = q
    if (DST / "config.json").is_symlink():
        (DST / "config.json").unlink()
    (DST / "config.json").write_text(json.dumps(cfg, indent=1))


def base_paths() -> None:
    """Fill templates/omlx/*.json (the exact files used for the published run, secrets removed)."""
    for kv in ("q8", "fp16"):
        base = ROOT / "engines" / f"omlx-base-{kv}"
        base.mkdir(parents=True, exist_ok=True)
        for name in ("settings", "model_settings"):
            text = (ROOT / "templates" / "omlx" / f"{name}-{kv}.json").read_text().replace("${ROOT}", str(ROOT))
            (base / f"{name}.json").write_text(text)


if __name__ == "__main__":
    model_view()
    base_paths()
    print("prepared", DST, "and engines/omlx-base-{q8,fp16}")
