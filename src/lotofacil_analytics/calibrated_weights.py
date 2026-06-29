from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


SUPERVISED_MODEL = "supervised_answer_key_calibration_v1"


def load_calibrated_weights(path: Path) -> Dict[str, float] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    weights = payload.get("weights") if isinstance(payload, dict) else None
    if not isinstance(weights, dict):
        return None
    out: Dict[str, float] = {}
    for key, value in weights.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out or None


def load_supervised_calibrated_weights(primary_path: Path, fallback_path: Path | None = None) -> Dict[str, float] | None:
    for path in [primary_path, fallback_path]:
        if path is None or not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("model") != SUPERVISED_MODEL:
            continue
        weights = payload.get("weights")
        if not isinstance(weights, dict):
            continue
        out: Dict[str, float] = {}
        for key, value in weights.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if out:
            return out
    return None
