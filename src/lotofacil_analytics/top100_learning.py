from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd

from .top50_refinement import REFINEMENT_FEATURES, normalize_weights


TOP100_WALKFORWARD_MODEL = "top100_walk_forward_learning_v1"


def load_top100_learning_payload(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("model") != TOP100_WALKFORWARD_MODEL:
        return None
    return payload


def _weighted_feature_score(row: pd.Series, weights: Mapping[str, float], *, inverse: bool = False) -> float:
    if not weights:
        return 50.0
    total = 0.0
    weight_sum = 0.0
    for feature, weight in weights.items():
        if feature not in row:
            continue
        value = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").fillna(50.0).iloc[0]
        score = 100.0 - float(value) if inverse else float(value)
        total += max(0.0, min(100.0, score)) * float(weight)
        weight_sum += float(weight)
    return round(total / weight_sum, 6) if weight_sum > 0.0 else 50.0


def apply_top100_learning(
    candidates: pd.DataFrame,
    payload: Mapping[str, object] | None,
    *,
    override_score_top100: bool = True,
) -> pd.DataFrame:
    out = candidates.copy()
    if out.empty:
        return out
    base = pd.to_numeric(out.get("score_top100", out.get("score_final", 50.0)), errors="coerce").fillna(50.0)
    out["score_top100_pre_aprendizado"] = base.round(6)
    out["score_aprendizado_top100"] = 50.0
    out["score_penalizador_top100"] = 50.0
    out["score_top100_aprendido"] = base.round(6)
    out["aprendizado_top100_aplicado"] = 0
    out["aprendizado_top100_model"] = ""
    if not payload:
        if override_score_top100:
            out["score_top100"] = out["score_top100_aprendido"]
        return out

    positive = normalize_weights(payload.get("positive_weights", {}) if isinstance(payload.get("positive_weights"), dict) else {})
    negative = normalize_weights(payload.get("negative_weights", {}) if isinstance(payload.get("negative_weights"), dict) else {})
    if not positive and not negative:
        if override_score_top100:
            out["score_top100"] = out["score_top100_aprendido"]
        return out

    base_weight = float(payload.get("base_weight", 0.62) or 0.62)
    base_weight = max(0.05, min(0.95, base_weight))
    learned_weight = 1.0 - base_weight

    learned_scores = []
    penalty_scores = []
    final_scores = []
    for _, row in out.iterrows():
        learned_score = _weighted_feature_score(row, positive, inverse=False)
        penalty_score = _weighted_feature_score(row, negative, inverse=True)
        blended = learned_score if not negative else ((learned_score * 0.60) + (penalty_score * 0.40))
        final = (float(row["score_top100_pre_aprendizado"]) * base_weight) + (blended * learned_weight)
        learned_scores.append(round(learned_score, 6))
        penalty_scores.append(round(penalty_score, 6))
        final_scores.append(round(max(0.0, min(100.0, final)), 6))

    out["score_aprendizado_top100"] = learned_scores
    out["score_penalizador_top100"] = penalty_scores
    out["score_top100_aprendido"] = final_scores
    out["aprendizado_top100_aplicado"] = 1
    out["aprendizado_top100_model"] = str(payload.get("model", TOP100_WALKFORWARD_MODEL))
    if override_score_top100:
        out["score_top100"] = out["score_top100_aprendido"]
    return out
