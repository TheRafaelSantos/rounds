from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd


TOP50_REFINEMENT_MODEL = "top50_refiner_post_error_walk_forward_v1"

REFINEMENT_FEATURES = (
    "score_top100",
    "score_final",
    "score_decisao_protegida",
    "score_estatistico",
    "score_historico",
    "score_atraso",
    "score_combinatorio",
    "score_localidade_numerologia",
    "score_contextual",
    "score_climatico",
    "score_temporal_profundo",
    "score_cenarios",
    "score_contrarian",
    "score_transicao",
    "score_cobertura_risco_falso_negativo",
    "score_combinatorio_avancado",
    "score_pares_atraso_freq",
    "score_trios_atraso_freq",
    "score_quartetos_atraso_freq",
    "score_grafo_dezenas",
    "score_complemento_ausentes",
    "score_geometria_volante",
    "score_residuos_modulares",
    "score_regimes_historicos",
    "score_detector_falso_positivo",
    "score_hard_negative",
    "score_learning_to_rank",
)


def normalize_weights(values: Mapping[str, float]) -> Dict[str, float]:
    cleaned = {
        str(key): max(0.0, float(value))
        for key, value in values.items()
        if str(key) in REFINEMENT_FEATURES
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        return {}
    return {key: round(value / total, 10) for key, value in sorted(cleaned.items())}


def load_top50_refinement_payload(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("model") != TOP50_REFINEMENT_MODEL:
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


def apply_top50_refinement(
    candidates: pd.DataFrame,
    payload: Mapping[str, object] | None,
    *,
    override_score_top100: bool = True,
) -> pd.DataFrame:
    out = candidates.copy()
    if out.empty:
        return out
    base = pd.to_numeric(out.get("score_top100", out.get("score_final", 50.0)), errors="coerce").fillna(50.0)
    out["score_top100_original"] = base.round(6)
    out["score_refinador_top50"] = 50.0
    out["score_penalizador_falso_top50"] = 50.0
    out["score_top50_refinado"] = base.round(6)
    out["refinador_top50_aplicado"] = 0
    out["refinador_top50_model"] = ""
    if not payload:
        if override_score_top100:
            out["score_top100"] = out["score_top50_refinado"]
        return out

    positive = normalize_weights(payload.get("positive_weights", {}) if isinstance(payload.get("positive_weights"), dict) else {})
    negative = normalize_weights(payload.get("negative_weights", {}) if isinstance(payload.get("negative_weights"), dict) else {})
    if not positive and not negative:
        if override_score_top100:
            out["score_top100"] = out["score_top50_refinado"]
        return out

    base_weight = float(payload.get("base_weight", 0.56) or 0.56)
    base_weight = max(0.05, min(0.95, base_weight))
    learned_weight = 1.0 - base_weight

    ref_scores = []
    penalty_scores = []
    final_scores = []
    for _, row in out.iterrows():
        ref_score = _weighted_feature_score(row, positive, inverse=False)
        penalty_score = _weighted_feature_score(row, negative, inverse=True)
        learned_score = ref_score if not negative else ((ref_score * 0.58) + (penalty_score * 0.42))
        final = (float(row["score_top100_original"]) * base_weight) + (learned_score * learned_weight)
        ref_scores.append(round(ref_score, 6))
        penalty_scores.append(round(penalty_score, 6))
        final_scores.append(round(max(0.0, min(100.0, final)), 6))

    out["score_refinador_top50"] = ref_scores
    out["score_penalizador_falso_top50"] = penalty_scores
    out["score_top50_refinado"] = final_scores
    out["refinador_top50_aplicado"] = 1
    out["refinador_top50_model"] = str(payload.get("model", TOP50_REFINEMENT_MODEL))
    if override_score_top100:
        out["score_top100"] = out["score_top50_refinado"]
    return out
