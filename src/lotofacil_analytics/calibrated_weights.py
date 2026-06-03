from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


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
