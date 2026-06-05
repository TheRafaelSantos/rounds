from __future__ import annotations

import json
import csv
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .backtest_lotofacil import compute_hits
from .exhaustive_optimizer import (
    DEFAULT_EXHAUSTIVE_WEIGHTS,
    EXHAUSTIVE_SOURCE_MODEL,
    FULL_EVEN_COUNT,
    FULL_SUM,
    NUMBERS,
    OMITTED_SIZE,
    PAIR_COUNT,
    TOTAL_COMBINATIONS,
    _climate_number_scores,
    _column_counts_from_omitted,
    _common_range_signatures,
    _contrarian_score,
    _context_number_scores,
    _delays,
    _diagonal_strength_from_omitted,
    _distance_outside_band,
    _historical_profile,
    _max_run_from_omitted,
    _pair_counter,
    _pair_selected_sum_from_omitted,
    _recent_freq,
    _scenario_score,
    _score_0_100,
    format_exhaustive_weights,
    resolve_exhaustive_weights,
)
from .context_features import build_context_model
from .normalize import DEZENAS
from .temporal_deep import temporal_deep_number_scores
from .transition_analysis import build_transition_model, score_transition_from_omitted
from .predictor import select_final_games
from .storage import sanitize_dataframe_for_tabular_output


WEIGHT_COMPONENTS = tuple(DEFAULT_EXHAUSTIVE_WEIGHTS.keys())
CACHE_SCHEMA_VERSION = 1
CACHE_SCORE_COMPONENTS = tuple(component for component in WEIGHT_COMPONENTS if component != "nao_repeticao_exata")
ATTEMPT_CACHE_COLUMNS = ["cache_status", "cache_rows", "cache_path"]
ELITE_MIN_HITS = 11
WEIGHT_STRATEGIES = {
    "preset",
    "random_exploration",
    "winner_average_mutation",
    "best_current_mutation",
    "elite_mutation",
    "elite_crossover",
    "elite_centroid",
}


@dataclass(frozen=True)
class CalibrationLabSummary:
    status: str
    current_concurso: int | None
    attempts_this_run: int
    total_attempts: int
    solved_contests: int
    best_hits_current: int
    best_game_current: str
    elapsed_seconds: float
    attempts_csv_path: str
    winners_csv_path: str
    state_json_path: str
    average_weights_csv_path: str
    engine_weights_json_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Calibracao 24/7",
                f"Status: {self.status}",
                f"Concurso em analise: {self.current_concurso if self.current_concurso is not None else 'nenhum'}",
                f"Tentativas nesta execucao: {self.attempts_this_run}",
                f"Tentativas totais registradas: {self.total_attempts}",
                f"Concursos resolvidos com 15 pontos: {self.solved_contests}",
                f"Melhor acerto do concurso atual: {self.best_hits_current}",
                f"Melhor jogo do concurso atual: {self.best_game_current or '-'}",
                f"Tempo desta execucao: {self.elapsed_seconds:.2f}s",
                f"CSV tentativas: {self.attempts_csv_path}",
                f"CSV concursos resolvidos: {self.winners_csv_path}",
                f"Estado retomavel: {self.state_json_path}",
                f"Media de pesos vencedores: {self.average_weights_csv_path}",
                f"Pesos aplicados no motor principal: {self.engine_weights_json_path}",
                "Mensagem: processo retomavel; se interromper, rode o mesmo comando para continuar.",
            ]
        )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def _parse_nums(text: str) -> List[int]:
    return sorted(int(part) for part in str(text).split())


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _parse_mask(text: object) -> int | None:
    try:
        nums = _parse_nums(str(text))
    except (TypeError, ValueError):
        return None
    if len(nums) != 15 or len(set(nums)) != 15:
        return None
    return _mask_from_selected(nums)


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except pd.errors.ParserError:
        df = _read_csv_flexible(path)
    return _repair_misaligned_attempt_columns(df)


def _append_csv(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([dict(row)])
    if not path.exists():
        new_row.to_csv(path, index=False, encoding="utf-8-sig")
        return

    existing = _read_csv(path)
    existing_cols = list(existing.columns)
    row_cols = list(new_row.columns)
    missing_in_existing = [column for column in row_cols if column not in existing_cols]
    missing_in_row = [column for column in existing_cols if column not in row_cols]
    if missing_in_existing or missing_in_row:
        columns = existing_cols + missing_in_existing
        merged = pd.concat(
            [
                existing.reindex(columns=columns),
                new_row.reindex(columns=columns),
            ],
            ignore_index=True,
        )
        merged.to_csv(path, index=False, encoding="utf-8-sig")
        return

    new_row.reindex(columns=existing_cols).to_csv(
        path,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8-sig",
    )


def _repair_misaligned_attempt_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [
        "score_weights",
        "peso_estatistico",
        "peso_historico",
        "peso_atraso",
        "peso_combinatorio",
        "peso_localidade_numerologia",
        "peso_climatico",
        "peso_temporal_profundo",
        "peso_cenarios",
        "peso_contrarian",
        "peso_transicao",
        "peso_nao_repeticao_exata",
        "weight_strategy",
        "elite_source_attempts",
        "elite_source_hits",
    ]
    if df.empty or any(column not in df.columns for column in ordered):
        return df
    marker = df["score_weights"].astype(str).isin(WEIGHT_STRATEGIES)
    if not bool(marker.any()):
        return df

    fixed = df.copy()
    for column in ordered:
        fixed[column] = fixed[column].astype(object)
    source = fixed.loc[marker, ordered].copy()
    mapping = {
        "weight_strategy": "score_weights",
        "elite_source_attempts": "peso_estatistico",
        "elite_source_hits": "peso_historico",
        "score_weights": "peso_atraso",
        "peso_estatistico": "peso_combinatorio",
        "peso_historico": "peso_localidade_numerologia",
        "peso_atraso": "peso_climatico",
        "peso_combinatorio": "peso_temporal_profundo",
        "peso_localidade_numerologia": "peso_cenarios",
        "peso_climatico": "peso_contrarian",
        "peso_temporal_profundo": "peso_transicao",
        "peso_cenarios": "peso_nao_repeticao_exata",
        "peso_contrarian": "weight_strategy",
        "peso_transicao": "elite_source_attempts",
        "peso_nao_repeticao_exata": "elite_source_hits",
    }
    for target, source_column in mapping.items():
        fixed.loc[marker, target] = source[source_column].to_numpy()
    return fixed


def _game_counts_for_target(attempts: pd.DataFrame, target_concurso: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    if attempts.empty or "target_concurso" not in attempts.columns:
        return counts
    df = attempts.copy()
    df["target_concurso"] = pd.to_numeric(df["target_concurso"], errors="coerce")
    df = df[df["target_concurso"] == int(target_concurso)].copy()
    if df.empty:
        return counts
    for column in ("jogo_1", "jogo_2"):
        if column not in df.columns:
            continue
        for value in df[column].dropna().tolist():
            mask = _parse_mask(value)
            if mask is not None:
                counts[int(mask)] += 1
    return counts


def _recent_game_masks_for_target(attempts: pd.DataFrame, target_concurso: int, *, recent_rows: int = 250) -> List[int]:
    if attempts.empty or "target_concurso" not in attempts.columns:
        return []
    df = attempts.copy()
    df["target_concurso"] = pd.to_numeric(df["target_concurso"], errors="coerce")
    df = df[df["target_concurso"] == int(target_concurso)].tail(int(recent_rows)).copy()
    masks: List[int] = []
    for column in ("jogo_1", "jogo_2"):
        if column not in df.columns:
            continue
        for value in df[column].dropna().tolist():
            mask = _parse_mask(value)
            if mask is not None:
                masks.append(int(mask))
    return masks


def _max_overlap_with_masks(mask: int, previous_masks: Sequence[int]) -> int:
    if not previous_masks:
        return 0
    return max(int(int(mask & previous).bit_count()) for previous in previous_masks)


def _novelty_penalty(*, repeat_count: int, recent_max_overlap: int) -> float:
    exact_penalty = min(24.0, float(max(0, repeat_count)) * 2.5)
    overlap_penalty = 0.0
    if int(recent_max_overlap) >= 15:
        overlap_penalty = 6.0
    elif int(recent_max_overlap) == 14:
        overlap_penalty = 3.5
    elif int(recent_max_overlap) == 13:
        overlap_penalty = 1.5
    elif int(recent_max_overlap) == 12:
        overlap_penalty = 0.5
    return round(float(exact_penalty + overlap_penalty), 6)


def _apply_calibration_novelty(
    candidates: pd.DataFrame,
    *,
    attempts: pd.DataFrame,
    target_concurso: int,
    recent_rows: int = 250,
) -> pd.DataFrame:
    if candidates.empty or "nums" not in candidates.columns or "score_final" not in candidates.columns:
        return candidates.copy()
    counts = _game_counts_for_target(attempts, target_concurso)
    recent_masks = _recent_game_masks_for_target(attempts, target_concurso, recent_rows=recent_rows)
    if not counts and not recent_masks:
        out = candidates.copy()
        out["score_final_original"] = pd.to_numeric(out["score_final"], errors="coerce")
        out["calibration_repeat_count"] = 0
        out["calibration_recent_max_overlap"] = 0
        out["calibration_novelty_penalty"] = 0.0
        out["score_calibration_novelty"] = 100.0
        return out

    rows: List[Dict[str, object]] = []
    for _, row in candidates.iterrows():
        out = row.to_dict()
        mask = _parse_mask(out.get("nums", ""))
        original_score = float(pd.to_numeric(pd.Series([out.get("score_final", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        repeat_count = int(counts.get(int(mask), 0)) if mask is not None else 0
        recent_overlap = _max_overlap_with_masks(int(mask), recent_masks) if mask is not None else 0
        penalty = _novelty_penalty(repeat_count=repeat_count, recent_max_overlap=recent_overlap)
        out["score_final_original"] = round(original_score, 6)
        out["calibration_repeat_count"] = repeat_count
        out["calibration_recent_max_overlap"] = int(recent_overlap)
        out["calibration_novelty_penalty"] = penalty
        out["score_calibration_novelty"] = round(max(0.0, 100.0 - (penalty * 3.0)), 6)
        out["score_final"] = round(max(0.0, original_score - penalty), 6)
        rows.append(out)
    return sanitize_dataframe_for_tabular_output(pd.DataFrame(rows))


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return pd.DataFrame()
    header = list(rows[0])
    data_rows = rows[1:]
    if not data_rows:
        return pd.DataFrame(columns=header)

    max_len = max(len(row) for row in data_rows)
    if "score_weights" in header and "cache_status" not in header and max_len == len(header) + len(ATTEMPT_CACHE_COLUMNS):
        score_idx = header.index("score_weights")
        migrated_header = header[:score_idx] + ATTEMPT_CACHE_COLUMNS + header[score_idx:]
        migrated_rows: List[List[str]] = []
        for row in data_rows:
            if len(row) == len(header):
                migrated_rows.append(row[:score_idx] + ["", "", ""] + row[score_idx:])
            else:
                migrated_rows.append((row + [""] * len(migrated_header))[: len(migrated_header)])
        return pd.DataFrame(migrated_rows, columns=migrated_header)

    if max_len > len(header):
        header = header + [f"extra_{idx}" for idx in range(1, max_len - len(header) + 1)]
    normalized = [(row + [""] * len(header))[: len(header)] for row in data_rows]
    return pd.DataFrame(normalized, columns=header)


def _target_climate_from_features(climate_features: pd.DataFrame | None, concurso: int) -> Mapping[str, object] | None:
    if climate_features is None or climate_features.empty or "concurso" not in climate_features.columns:
        return None
    df = climate_features.copy()
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")
    rows = df[df["concurso"] == int(concurso)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _delete_if_exists(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _solved_contests(winners: pd.DataFrame) -> set[int]:
    if winners.empty or "target_concurso" not in winners.columns:
        return set()
    values = pd.to_numeric(winners["target_concurso"], errors="coerce").dropna()
    return set(int(value) for value in values)


def _next_target(contests: Sequence[int], solved: set[int], preferred: int | None) -> int | None:
    ordered = [int(value) for value in contests if int(value) not in solved]
    if not ordered:
        return None
    if preferred is not None and int(preferred) in ordered:
        return int(preferred)
    return int(ordered[0])


def _last_attempt_for_target(attempts: pd.DataFrame, target_concurso: int) -> int:
    if attempts.empty or "target_concurso" not in attempts.columns or "tentativa" not in attempts.columns:
        return 0
    target_rows = attempts[pd.to_numeric(attempts["target_concurso"], errors="coerce") == int(target_concurso)]
    if target_rows.empty:
        return 0
    values = pd.to_numeric(target_rows["tentativa"], errors="coerce").dropna()
    return int(values.max()) if len(values) else 0


def _best_for_target(attempts: pd.DataFrame, target_concurso: int) -> Dict[str, object]:
    if attempts.empty or "target_concurso" not in attempts.columns:
        return {"hits": 0, "game": "", "attempt": 0, "weights": {}}
    rows = attempts[pd.to_numeric(attempts["target_concurso"], errors="coerce") == int(target_concurso)].copy()
    if rows.empty or "melhor_acerto" not in rows.columns:
        return {"hits": 0, "game": "", "attempt": 0, "weights": {}}
    rows["melhor_acerto_numeric"] = pd.to_numeric(rows["melhor_acerto"], errors="coerce").fillna(0)
    rows["tentativa_numeric"] = pd.to_numeric(rows["tentativa"], errors="coerce").fillna(0)
    best = rows.sort_values(["melhor_acerto_numeric", "tentativa_numeric"], ascending=[False, True]).iloc[0]
    weights: Dict[str, float] = {}
    for component in WEIGHT_COMPONENTS:
        column = f"peso_{component}"
        if column in best and pd.notna(best[column]):
            weights[component] = float(best[column])
    return {
        "hits": int(best["melhor_acerto_numeric"]),
        "game": str(best.get("melhor_jogo", "")),
        "attempt": int(best["tentativa_numeric"]),
        "weights": weights,
    }


def _weights_from_row(row: pd.Series | Mapping[str, object]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for component in WEIGHT_COMPONENTS:
        column = f"peso_{component}"
        value = row.get(column) if isinstance(row, Mapping) else row.get(column)
        if value is None or pd.isna(value):
            continue
        try:
            weights[component] = float(value)
        except (TypeError, ValueError):
            continue
    return resolve_exhaustive_weights(weights) if weights else {}


def _elite_attempts_from(attempts: pd.DataFrame, *, min_hits: int = ELITE_MIN_HITS) -> pd.DataFrame:
    if attempts.empty or "melhor_acerto" not in attempts.columns:
        return pd.DataFrame()
    df = attempts.copy()
    df["melhor_acerto"] = pd.to_numeric(df["melhor_acerto"], errors="coerce").fillna(0).astype(int)
    df = df[df["melhor_acerto"] >= int(min_hits)].copy()
    if df.empty:
        return df
    df["elite_level"] = df["melhor_acerto"].map(lambda value: f"elite_{int(value)}")
    df["elite_saved_at"] = _now()
    keep_columns = [
        "target_concurso",
        "data_sorteio",
        "tentativa",
        "generated_at",
        "elite_saved_at",
        "elite_level",
        "melhor_acerto",
        "jogo_1",
        "acertos_jogo_1",
        "jogo_2",
        "acertos_jogo_2",
        "melhor_jogo",
        "score_jogo_1",
        "score_jogo_2",
        "score_original_jogo_1",
        "score_original_jogo_2",
        "repeat_count_jogo_1",
        "repeat_count_jogo_2",
        "recent_overlap_jogo_1",
        "recent_overlap_jogo_2",
        "novelty_penalty_jogo_1",
        "novelty_penalty_jogo_2",
        "score_novelty_jogo_1",
        "score_novelty_jogo_2",
        "weight_strategy",
        "elite_source_attempts",
        "elite_source_hits",
        "score_weights",
    ]
    keep_columns.extend(f"peso_{component}" for component in WEIGHT_COMPONENTS)
    for column in keep_columns:
        if column not in df.columns:
            df[column] = ""
    return df[keep_columns].copy()


def _sync_elites_from_attempts(
    *,
    attempts: pd.DataFrame,
    elites_csv_path: Path,
    min_hits: int = ELITE_MIN_HITS,
) -> pd.DataFrame:
    existing = _read_csv(elites_csv_path)
    discovered = _elite_attempts_from(attempts, min_hits=min_hits)
    if existing.empty and discovered.empty:
        return pd.DataFrame()
    merged = pd.concat([existing, discovered], ignore_index=True)
    if merged.empty:
        return merged
    for column in ["target_concurso", "tentativa", "melhor_acerto"]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged = merged.dropna(subset=["target_concurso", "tentativa"]).copy()
    if merged.empty:
        return merged
    merged["target_concurso"] = merged["target_concurso"].astype(int)
    merged["tentativa"] = merged["tentativa"].astype(int)
    merged["melhor_acerto"] = pd.to_numeric(merged["melhor_acerto"], errors="coerce").fillna(0).astype(int)
    merged = merged.sort_values(
        ["target_concurso", "melhor_acerto", "tentativa"],
        ascending=[True, False, True],
    ).drop_duplicates(["target_concurso", "tentativa"], keep="first")
    merged = merged.sort_values(["target_concurso", "melhor_acerto", "tentativa"], ascending=[True, False, True])
    elites_csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(elites_csv_path, index=False, encoding="utf-8-sig")
    return merged.reset_index(drop=True)


def _elite_rows_for_target(elites: pd.DataFrame, target_concurso: int) -> pd.DataFrame:
    if elites.empty or "target_concurso" not in elites.columns:
        return pd.DataFrame()
    df = elites.copy()
    df["target_concurso"] = pd.to_numeric(df["target_concurso"], errors="coerce")
    df["melhor_acerto"] = pd.to_numeric(df.get("melhor_acerto", 0), errors="coerce").fillna(0).astype(int)
    df = df[df["target_concurso"] == int(target_concurso)].copy()
    if df.empty:
        return df
    return df.sort_values(["melhor_acerto", "tentativa"], ascending=[False, True]).reset_index(drop=True)


def _elite_stats(elites: pd.DataFrame, target_concurso: int) -> Dict[str, object]:
    target_elites = _elite_rows_for_target(elites, target_concurso)
    if target_elites.empty:
        return {
            "elite_count_current": 0,
            "elite_best_hits_current": 0,
            "elite_best_attempt_current": 0,
            "elite_counts_by_hits_current": {},
        }
    counts = {
        str(int(hits)): int(count)
        for hits, count in target_elites["melhor_acerto"].value_counts().sort_index().items()
    }
    best = target_elites.iloc[0]
    return {
        "elite_count_current": int(len(target_elites)),
        "elite_best_hits_current": int(best["melhor_acerto"]),
        "elite_best_attempt_current": int(best.get("tentativa", 0)),
        "elite_counts_by_hits_current": counts,
    }


def _select_elite_row(elites: pd.DataFrame, rng: random.Random) -> pd.Series:
    if elites.empty:
        raise ValueError("Nenhuma elite disponivel para selecao.")
    rows = elites.copy().reset_index(drop=True)
    hits = pd.to_numeric(rows["melhor_acerto"], errors="coerce").fillna(0)
    attempts = pd.to_numeric(rows.get("tentativa", 0), errors="coerce").fillna(0)
    max_attempt = float(attempts.max()) if len(attempts) else 0.0
    recency = 1.0 + ((attempts / max(1.0, max_attempt)) * 0.35)
    score = ((hits - 10).clip(lower=1) ** 3) * recency
    weights = [float(value) for value in score.tolist()]
    index = rng.choices(list(range(len(rows))), weights=weights, k=1)[0]
    return rows.iloc[int(index)]


def _mutation_sigma(best_hits: int) -> float:
    if int(best_hits) >= 14:
        return 0.08
    if int(best_hits) >= 13:
        return 0.14
    if int(best_hits) >= 12:
        return 0.22
    return 0.34


def _mutate_weight_anchor(anchor: Mapping[str, float], *, rng: random.Random, best_hits: int, zero_probability: float | None = None) -> Dict[str, float]:
    sigma = _mutation_sigma(best_hits)
    zero_prob = zero_probability if zero_probability is not None else (0.005 if best_hits >= 13 else 0.02)
    values: Dict[str, float] = {}
    for component in WEIGHT_COMPONENTS:
        base = max(0.0001, float(anchor.get(component, DEFAULT_EXHAUSTIVE_WEIGHTS[component])))
        values[component] = base * rng.lognormvariate(0.0, sigma)
        if rng.random() < zero_prob:
            values[component] = 0.0
    return resolve_exhaustive_weights(values)


def _average_elite_weights(elites: pd.DataFrame, limit: int = 12) -> Dict[str, float]:
    if elites.empty:
        return {}
    ranked = elites.sort_values(["melhor_acerto", "tentativa"], ascending=[False, True]).head(int(limit)).copy()
    totals = {component: 0.0 for component in WEIGHT_COMPONENTS}
    total_weight = 0.0
    for _, row in ranked.iterrows():
        weights = _weights_from_row(row)
        if not weights:
            continue
        hit_weight = max(1.0, float(row.get("melhor_acerto", 0)) - 10.0) ** 2
        total_weight += hit_weight
        for component in WEIGHT_COMPONENTS:
            totals[component] += float(weights.get(component, 0.0)) * hit_weight
    if total_weight <= 0:
        return {}
    return resolve_exhaustive_weights({component: value / total_weight for component, value in totals.items()})


def _crossover_elite_weights(elites: pd.DataFrame, rng: random.Random) -> Tuple[Dict[str, float], str, str]:
    first = _select_elite_row(elites, rng)
    second = _select_elite_row(elites, rng)
    first_weights = _weights_from_row(first)
    second_weights = _weights_from_row(second)
    if not first_weights or not second_weights:
        return {}, "", ""
    alpha = rng.uniform(0.35, 0.65)
    values = {
        component: (alpha * float(first_weights.get(component, 0.0)))
        + ((1.0 - alpha) * float(second_weights.get(component, 0.0)))
        for component in WEIGHT_COMPONENTS
    }
    best_hits = max(int(first.get("melhor_acerto", 0)), int(second.get("melhor_acerto", 0)))
    crossed = _mutate_weight_anchor(resolve_exhaustive_weights(values), rng=rng, best_hits=best_hits, zero_probability=0.0)
    attempts = f"{int(first.get('tentativa', 0))}|{int(second.get('tentativa', 0))}"
    hits = f"{int(first.get('melhor_acerto', 0))}|{int(second.get('melhor_acerto', 0))}"
    return crossed, attempts, hits


def _average_winner_weights(winners: pd.DataFrame) -> Dict[str, float] | None:
    if winners.empty:
        return None
    means: Dict[str, float] = {}
    for component in WEIGHT_COMPONENTS:
        column = f"peso_{component}"
        if column not in winners.columns:
            continue
        values = pd.to_numeric(winners[column], errors="coerce").dropna()
        if len(values):
            means[component] = float(values.mean())
    if not means:
        return None
    return resolve_exhaustive_weights(means)


def _write_average_outputs(
    *,
    winners: pd.DataFrame,
    average_weights_csv_path: Path,
    engine_weights_json_path: Path,
) -> Dict[str, float] | None:
    average = _average_winner_weights(winners)
    average_weights_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if average is None:
        pd.DataFrame(columns=["componente", "peso", "peso_percentual"]).to_csv(
            average_weights_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        return None

    rows = [
        {
            "componente": component,
            "peso": round(float(average[component]), 10),
            "peso_percentual": round(float(average[component]) * 100.0, 6),
        }
        for component in WEIGHT_COMPONENTS
    ]
    pd.DataFrame(rows).to_csv(average_weights_csv_path, index=False, encoding="utf-8-sig")
    contests = []
    if "target_concurso" in winners.columns:
        contests = [
            int(value)
            for value in pd.to_numeric(winners["target_concurso"], errors="coerce").dropna().tolist()
        ]
    payload = {
        "source": "calibration_lab_winners_average_v1",
        "updated_at": _now(),
        "solved_contests": contests,
        "solved_count": int(len(contests)),
        "weights": {component: round(float(average[component]), 10) for component in WEIGHT_COMPONENTS},
        "score_weights": format_exhaustive_weights(average),
        "note": "Media dos pesos das tentativas que acertaram 15 dezenas em um dos dois jogos.",
    }
    _write_json(engine_weights_json_path, payload)
    return average


def _write_summary(
    *,
    summary_csv_path: Path,
    state: Mapping[str, object],
    attempts: pd.DataFrame,
    winners: pd.DataFrame,
    elites: pd.DataFrame | None = None,
) -> None:
    rows = [
        {"metrica": "status", "valor": state.get("status", "")},
        {"metrica": "current_concurso", "valor": state.get("current_concurso", "")},
        {"metrica": "current_attempt", "valor": state.get("current_attempt", "")},
        {"metrica": "total_attempts", "valor": int(len(attempts))},
        {"metrica": "solved_contests", "valor": int(len(_solved_contests(winners)))},
        {"metrica": "elites_total", "valor": int(0 if elites is None else len(elites))},
        {"metrica": "best_hits_current", "valor": state.get("best_hits_current", 0)},
        {"metrica": "elite_best_hits_current", "valor": state.get("elite_best_hits_current", 0)},
        {"metrica": "best_game_current", "valor": state.get("best_game_current", "")},
        {"metrica": "updated_at", "valor": state.get("updated_at", "")},
    ]
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_csv_path, index=False, encoding="utf-8-sig")


def _write_excel_snapshot(
    *,
    excel_path: Path,
    attempts: pd.DataFrame,
    winners: pd.DataFrame,
    elites: pd.DataFrame,
    average_weights_csv_path: Path,
    summary_csv_path: Path,
) -> None:
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(summary_csv_path)
    average = _read_csv(average_weights_csv_path)
    attempts_tail = attempts.tail(500).copy() if not attempts.empty else attempts
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="resumo")
        average.to_excel(writer, index=False, sheet_name="pesos_medios")
        winners.to_excel(writer, index=False, sheet_name="vencedores")
        elites.tail(1000).to_excel(writer, index=False, sheet_name="elites")
        attempts_tail.to_excel(writer, index=False, sheet_name="ultimas_tentativas")


def _mask_from_selected(selected: Sequence[int]) -> int:
    mask = 0
    for n in selected:
        mask |= 1 << (int(n) - 1)
    return int(mask)


def _nums_from_mask(mask: int) -> List[int]:
    return [n for n in NUMBERS if int(mask) & (1 << (int(n) - 1))]


def _format_mask(mask: int) -> str:
    return _format_nums(_nums_from_mask(mask))


def _cache_paths(cache_dir: Path, target_concurso: int) -> Dict[str, Path]:
    base = cache_dir / f"concurso_{int(target_concurso)}"
    return {
        "base": base,
        "meta": base / "meta.json",
        "masks": base / "masks.npy",
        "scores": base / "scores.npy",
        "soma": base / "soma.npy",
        "qtd_pares": base / "qtd_pares.npy",
        "overlap_ultimo": base / "overlap_ultimo.npy",
        "maior_sequencia": base / "maior_sequencia.npy",
        "ja_saiu": base / "ja_saiu.npy",
    }


def _cache_meta_matches(meta: Mapping[str, object], *, target_concurso: int, train_df: pd.DataFrame, exhaustive_limit: int | None, draw_hour: int, draw_minute: int) -> bool:
    if not meta:
        return False
    expected_rows = int(TOTAL_COMBINATIONS if exhaustive_limit is None else min(int(exhaustive_limit), int(TOTAL_COMBINATIONS)))
    try:
        return (
            int(meta.get("schema_version", 0)) == CACHE_SCHEMA_VERSION
            and int(meta.get("target_concurso", 0)) == int(target_concurso)
            and int(meta.get("concurso_base_final", 0)) == int(pd.to_numeric(train_df["concurso"], errors="coerce").max())
            and int(meta.get("draw_hour", -1)) == int(draw_hour)
            and int(meta.get("draw_minute", -1)) == int(draw_minute)
            and meta.get("exhaustive_limit") == (int(exhaustive_limit) if exhaustive_limit is not None else None)
            and int(meta.get("rows", 0)) == expected_rows
            and list(meta.get("score_components", [])) == list(CACHE_SCORE_COMPONENTS)
        )
    except (TypeError, ValueError):
        return False


def _cache_is_complete(paths: Mapping[str, Path], *, expected_rows: int) -> bool:
    required = ["meta", "masks", "scores", "soma", "qtd_pares", "overlap_ultimo", "maior_sequencia", "ja_saiu"]
    if any(not paths[name].exists() for name in required):
        return False
    try:
        scores = np.load(paths["scores"], mmap_mode="r")
        masks = np.load(paths["masks"], mmap_mode="r")
        return int(scores.shape[0]) == int(expected_rows) and int(masks.shape[0]) == int(expected_rows)
    except (OSError, ValueError):
        return False


def _cleanup_old_caches(cache_dir: Path, *, keep_concurso: int, keep_last: int = 1) -> None:
    if not cache_dir.exists():
        return
    cache_dirs = [path for path in cache_dir.iterdir() if path.is_dir() and path.name.startswith("concurso_")]
    parsed: List[Tuple[int, Path]] = []
    for path in cache_dirs:
        try:
            parsed.append((int(path.name.split("_", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    protected = {int(keep_concurso)}
    if keep_last > 1:
        for concurso, _path in sorted(parsed, reverse=True)[: int(keep_last)]:
            protected.add(int(concurso))
    for concurso, path in parsed:
        if int(concurso) in protected:
            continue
        for child in path.glob("*"):
            if child.is_file():
                child.unlink()
        try:
            path.rmdir()
        except OSError:
            pass


def _build_component_cache(
    *,
    train_df: pd.DataFrame,
    target_concurso: int,
    climate_features: pd.DataFrame | None,
    exhaustive_limit: int | None,
    draw_hour: int,
    draw_minute: int,
    cache_dir: Path,
) -> Dict[str, object]:
    paths = _cache_paths(cache_dir, target_concurso)
    expected_rows = int(TOTAL_COMBINATIONS if exhaustive_limit is None else min(int(exhaustive_limit), int(TOTAL_COMBINATIONS)))
    meta = _load_json(paths["meta"])
    if _cache_meta_matches(
        meta,
        target_concurso=target_concurso,
        train_df=train_df,
        exhaustive_limit=exhaustive_limit,
        draw_hour=draw_hour,
        draw_minute=draw_minute,
    ) and _cache_is_complete(paths, expected_rows=expected_rows):
        return {"status": "hit", "rows": expected_rows, "cache_path": str(paths["base"])}

    if paths["base"].exists():
        for child in paths["base"].glob("*"):
            if child.is_file():
                child.unlink()
    paths["base"].mkdir(parents=True, exist_ok=True)
    _cleanup_old_caches(cache_dir, keep_concurso=target_concurso)

    df = train_df.copy().sort_values("concurso").reset_index(drop=True)
    draws = [_nums_from_row(row) for _, row in df.iterrows()]
    profile = _historical_profile(draws)
    last_draw = set(draws[-1])
    recent_freq = _recent_freq(draws, window=100)
    delays = _delays(draws)
    pair_freq = _pair_counter(draws)
    common_signatures = _common_range_signatures(draws)
    context_model = build_context_model(
        df,
        draw_hour=draw_hour,
        draw_minute=draw_minute,
        climate_features=climate_features,
        target_climate=_target_climate_from_features(climate_features, target_concurso),
    )
    context_scores = _context_number_scores(context_model)
    climate_scores = _climate_number_scores(context_model)
    temporal_deep_scores = temporal_deep_number_scores(df, target_date=context_model.target.data_proximo_concurso)
    transition_model = build_transition_model(df)

    recent_total = sum(float(recent_freq.get(n, 0)) for n in NUMBERS)
    delay_total = sum(float(delays.get(n, 0)) for n in NUMBERS)
    context_total = sum(float(context_scores.get(n, 50.0)) for n in NUMBERS)
    climate_total = sum(float(climate_scores.get(n, 50.0)) for n in NUMBERS)
    temporal_deep_total = sum(float(temporal_deep_scores.get(n, 50.0)) for n in NUMBERS)
    pair_values = list(pair_freq.values())
    pair_median = float(pd.Series(pair_values).median()) if pair_values else 0.0
    pair_matrix = [[0.0 for _ in range(max(NUMBERS) + 1)] for _ in range(max(NUMBERS) + 1)]
    for (a, b), value in pair_freq.items():
        pair_matrix[int(a)][int(b)] = float(value)
        pair_matrix[int(b)][int(a)] = float(value)
    total_pair_sum = sum(pair_matrix[a][b] for a, b in combinations(NUMBERS, 2))
    incident_pair_sum = {
        n: sum(pair_matrix[n][other] for other in NUMBERS if other != n)
        for n in NUMBERS
    }
    delay_series = pd.Series(list(delays.values()), dtype="float64")
    median_delay = float(delay_series.median()) if len(delay_series) else 0.0
    existing_draws = {tuple(draw) for draw in draws}

    masks = np.zeros(expected_rows, dtype=np.uint32)
    scores = np.zeros((expected_rows, len(CACHE_SCORE_COMPONENTS)), dtype=np.float32)
    soma = np.zeros(expected_rows, dtype=np.uint16)
    qtd_pares = np.zeros(expected_rows, dtype=np.uint8)
    overlap_ultimo = np.zeros(expected_rows, dtype=np.uint8)
    maior_sequencia = np.zeros(expected_rows, dtype=np.uint8)
    ja_saiu = np.zeros(expected_rows, dtype=np.uint8)

    evaluated = 0
    component_index = {component: idx for idx, component in enumerate(CACHE_SCORE_COMPONENTS)}
    for omitted in combinations(NUMBERS, OMITTED_SIZE):
        if evaluated >= expected_rows:
            break
        omitted_set = set(int(n) for n in omitted)
        selected = tuple(n for n in NUMBERS if n not in omitted_set)
        selected_sum = FULL_SUM - sum(omitted)
        selected_pairs = FULL_EVEN_COUNT - sum(1 for n in omitted if n % 2 == 0)
        ranges = [5, 5, 5, 5, 5]
        for n in omitted:
            ranges[(int(n) - 1) // 5] -= 1
        columns = _column_counts_from_omitted(omitted)
        max_run = _max_run_from_omitted(omitted_set)
        overlap_last = len(set(selected) & last_draw)
        diagonal_strength = _diagonal_strength_from_omitted(omitted_set)
        exact_historical = tuple(selected) in existing_draws

        signature = "-".join(str(value) for value in ranges)
        stat_penalty = 0.0
        stat_penalty += _distance_outside_band(selected_sum, profile["sum_p10"], profile["sum_p90"]) / 1.6
        stat_penalty += _distance_outside_band(selected_pairs, profile["pairs_p10"], profile["pairs_p90"]) * 3.0
        stat_penalty += 0.0 if signature in common_signatures else sum(abs(value - 3) for value in ranges) * 1.8
        stat_penalty += abs(overlap_last - profile["median_overlap"]) * 2.2
        stat_penalty += max(0.0, max_run - profile["run_p95"]) * 2.2
        score_estatistico = _score_0_100(stat_penalty)

        selected_recent_sum = recent_total - sum(float(recent_freq.get(int(n), 0)) for n in omitted)
        avg_recent = selected_recent_sum / len(selected)
        expected_recent = 100 * len(selected) / len(NUMBERS)
        score_historico = _score_0_100(abs(avg_recent - expected_recent) * 1.2)

        selected_delay_sum = delay_total - sum(float(delays.get(int(n), 0)) for n in omitted)
        avg_delay = selected_delay_sum / len(selected)
        score_atraso = _score_0_100(abs(avg_delay - median_delay) * 1.4)

        selected_pair_sum = _pair_selected_sum_from_omitted(
            total_pair_sum=total_pair_sum,
            incident_pair_sum=incident_pair_sum,
            pair_matrix=pair_matrix,
            omitted=omitted,
        )
        avg_pair_freq = selected_pair_sum / PAIR_COUNT
        score_combinatorio = _score_0_100(max(0.0, avg_pair_freq - pair_median) / 12.0)

        selected_context_sum = context_total - sum(float(context_scores.get(int(n), 50.0)) for n in omitted)
        score_contextual = round(max(0.0, min(100.0, selected_context_sum / len(selected))), 6)
        selected_climate_sum = climate_total - sum(float(climate_scores.get(int(n), 50.0)) for n in omitted)
        score_climatico = round(max(0.0, min(100.0, selected_climate_sum / len(selected))), 6)
        selected_temporal_deep_sum = temporal_deep_total - sum(float(temporal_deep_scores.get(int(n), 50.0)) for n in omitted)
        score_temporal_profundo = round(max(0.0, min(100.0, selected_temporal_deep_sum / len(selected))), 6)
        score_cenarios = _scenario_score(
            total=selected_sum,
            max_run=max_run,
            ranges=ranges,
            columns=columns,
            diagonal_strength=diagonal_strength,
            profile=profile,
            common_signatures=common_signatures,
            selected=selected,
        )
        score_contrarian = _contrarian_score(selected, max_run=max_run, ranges=ranges, columns=columns)
        transition_detail = score_transition_from_omitted(omitted, transition_model)

        masks[evaluated] = _mask_from_selected(selected)
        scores[evaluated, component_index["estatistico"]] = score_estatistico
        scores[evaluated, component_index["historico"]] = score_historico
        scores[evaluated, component_index["atraso"]] = score_atraso
        scores[evaluated, component_index["combinatorio"]] = score_combinatorio
        scores[evaluated, component_index["localidade_numerologia"]] = score_contextual
        scores[evaluated, component_index["climatico"]] = score_climatico
        scores[evaluated, component_index["temporal_profundo"]] = score_temporal_profundo
        scores[evaluated, component_index["cenarios"]] = score_cenarios
        scores[evaluated, component_index["contrarian"]] = score_contrarian
        scores[evaluated, component_index["transicao"]] = float(transition_detail["score_transicao"])
        soma[evaluated] = int(selected_sum)
        qtd_pares[evaluated] = int(selected_pairs)
        overlap_ultimo[evaluated] = int(overlap_last)
        maior_sequencia[evaluated] = int(max_run)
        ja_saiu[evaluated] = int(exact_historical)
        evaluated += 1

    if evaluated != expected_rows:
        masks = masks[:evaluated]
        scores = scores[:evaluated]
        soma = soma[:evaluated]
        qtd_pares = qtd_pares[:evaluated]
        overlap_ultimo = overlap_ultimo[:evaluated]
        maior_sequencia = maior_sequencia[:evaluated]
        ja_saiu = ja_saiu[:evaluated]

    np.save(paths["masks"], masks)
    np.save(paths["scores"], scores)
    np.save(paths["soma"], soma)
    np.save(paths["qtd_pares"], qtd_pares)
    np.save(paths["overlap_ultimo"], overlap_ultimo)
    np.save(paths["maior_sequencia"], maior_sequencia)
    np.save(paths["ja_saiu"], ja_saiu)

    meta_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at": _now(),
        "target_concurso": int(target_concurso),
        "rows": int(evaluated),
        "score_components": list(CACHE_SCORE_COMPONENTS),
        "exhaustive_limit": int(exhaustive_limit) if exhaustive_limit is not None else None,
        "draw_hour": int(draw_hour),
        "draw_minute": int(draw_minute),
        "concurso_base_inicial": int(pd.to_numeric(df["concurso"], errors="coerce").min()),
        "concurso_base_final": int(pd.to_numeric(df["concurso"], errors="coerce").max()),
        "source_model": EXHAUSTIVE_SOURCE_MODEL,
    }
    _write_json(paths["meta"], meta_payload)
    return {"status": "built", "rows": int(evaluated), "cache_path": str(paths["base"])}


def _load_cached_top_candidates(
    *,
    target_concurso: int,
    cache_dir: Path,
    weights: Mapping[str, float],
    top_games: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    paths = _cache_paths(cache_dir, target_concurso)
    meta = _load_json(paths["meta"])
    if not meta:
        raise ValueError(f"Cache do concurso {target_concurso} nao encontrado.")

    masks = np.load(paths["masks"], mmap_mode="r")
    component_scores = np.load(paths["scores"], mmap_mode="r")
    ja_saiu = np.load(paths["ja_saiu"], mmap_mode="r")
    resolved = resolve_exhaustive_weights(weights)
    vector = np.array([float(resolved[component]) for component in CACHE_SCORE_COMPONENTS], dtype=np.float32)
    score_final = np.asarray(component_scores @ vector, dtype=np.float32)
    repetition_score = np.where(ja_saiu.astype(np.uint8) == 1, 92.0, 100.0).astype(np.float32)
    score_final = score_final + (float(resolved["nao_repeticao_exata"]) * repetition_score)

    keep = max(2, min(int(top_games), int(score_final.shape[0])))
    if keep >= int(score_final.shape[0]):
        top_indices = np.arange(int(score_final.shape[0]))
    else:
        top_indices = np.argpartition(score_final, -keep)[-keep:]
    top_masks = np.asarray(masks[top_indices], dtype=np.uint32)
    order = np.lexsort((top_masks, -score_final[top_indices]))
    ranked_indices = top_indices[order]

    soma = np.load(paths["soma"], mmap_mode="r")
    qtd_pares = np.load(paths["qtd_pares"], mmap_mode="r")
    overlap_ultimo = np.load(paths["overlap_ultimo"], mmap_mode="r")
    maior_sequencia = np.load(paths["maior_sequencia"], mmap_mode="r")
    rows: List[Dict[str, object]] = []
    for rank, idx in enumerate(ranked_indices.tolist(), start=1):
        row: Dict[str, object] = {
            "rank": int(rank),
            "nums": _format_mask(int(masks[idx])),
            "source_model": EXHAUSTIVE_SOURCE_MODEL,
            "metodo": EXHAUSTIVE_SOURCE_MODEL,
            "score_final": round(float(score_final[idx]), 6),
            "soma": int(soma[idx]),
            "qtd_pares": int(qtd_pares[idx]),
            "overlap_ultimo": int(overlap_ultimo[idx]),
            "maior_sequencia": int(maior_sequencia[idx]),
            "ja_saiu_exatamente_no_historico": int(ja_saiu[idx]),
            "total_combinacoes_avaliadas": int(meta.get("rows", len(masks))),
            "concurso_base_inicial": int(meta.get("concurso_base_inicial", 0)),
            "concurso_base_final": int(meta.get("concurso_base_final", 0)),
        }
        for component_idx, component in enumerate(CACHE_SCORE_COMPONENTS):
            value = float(component_scores[idx, component_idx])
            row[f"score_{component}"] = round(value, 6)
            if component == "localidade_numerologia":
                row["score_contextual"] = round(value, 6)
        rows.append(row)

    candidates = sanitize_dataframe_for_tabular_output(pd.DataFrame(rows))
    cache_info = {
        "status": "ranked",
        "rows": int(meta.get("rows", len(masks))),
        "cache_path": str(paths["base"]),
    }
    return candidates, cache_info


def _base_weight_presets() -> List[Dict[str, float]]:
    return [
        dict(DEFAULT_EXHAUSTIVE_WEIGHTS),
        {**DEFAULT_EXHAUSTIVE_WEIGHTS, "localidade_numerologia": 0.0},
        {**DEFAULT_EXHAUSTIVE_WEIGHTS, "climatico": 0.0},
        {**DEFAULT_EXHAUSTIVE_WEIGHTS, "temporal_profundo": 0.0},
        {
            **DEFAULT_EXHAUSTIVE_WEIGHTS,
            "localidade_numerologia": 0.0,
            "climatico": 0.0,
            "temporal_profundo": 0.0,
        },
        {
            **DEFAULT_EXHAUSTIVE_WEIGHTS,
            "combinatorio": 0.22,
            "transicao": 0.20,
            "historico": 0.16,
            "localidade_numerologia": 0.04,
        },
        {
            **DEFAULT_EXHAUSTIVE_WEIGHTS,
            "contrarian": 0.22,
            "cenarios": 0.18,
            "nao_repeticao_exata": 0.10,
            "localidade_numerologia": 0.04,
        },
        {
            **DEFAULT_EXHAUSTIVE_WEIGHTS,
            "temporal_profundo": 0.18,
            "historico": 0.18,
            "atraso": 0.12,
            "localidade_numerologia": 0.05,
        },
    ]


def _weights_for_attempt(
    *,
    target_concurso: int,
    attempt: int,
    seed: int,
    average_winner_weights: Mapping[str, float] | None,
    best_current_weights: Mapping[str, float] | None,
    elite_rows: pd.DataFrame | None = None,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    presets = _base_weight_presets()
    rng = random.Random(int(seed) + int(target_concurso) * 1000003 + int(attempt) * 7919)
    target_elites = elite_rows.copy() if elite_rows is not None and not elite_rows.empty else pd.DataFrame()
    if not target_elites.empty:
        roll = rng.random()
        if roll < 0.60:
            elite = _select_elite_row(target_elites, rng)
            anchor = _weights_from_row(elite)
            if anchor:
                hits = int(elite.get("melhor_acerto", 0))
                weights = _mutate_weight_anchor(anchor, rng=rng, best_hits=hits)
                return weights, {
                    "weight_strategy": "elite_mutation",
                    "elite_source_attempts": str(int(elite.get("tentativa", 0))),
                    "elite_source_hits": str(hits),
                }
        if roll < 0.82 and len(target_elites) >= 2:
            crossed, attempts_label, hits_label = _crossover_elite_weights(target_elites, rng)
            if crossed:
                return crossed, {
                    "weight_strategy": "elite_crossover",
                    "elite_source_attempts": attempts_label,
                    "elite_source_hits": hits_label,
                }
        if roll < 0.95:
            anchor = _average_elite_weights(target_elites)
            if anchor:
                best_hits = int(pd.to_numeric(target_elites["melhor_acerto"], errors="coerce").max())
                weights = _mutate_weight_anchor(anchor, rng=rng, best_hits=best_hits, zero_probability=0.0)
                return weights, {
                    "weight_strategy": "elite_centroid",
                    "elite_source_attempts": "top_elites",
                    "elite_source_hits": str(best_hits),
                }

    if int(attempt) <= len(presets):
        return resolve_exhaustive_weights(presets[int(attempt) - 1]), {
            "weight_strategy": "preset",
            "elite_source_attempts": "",
            "elite_source_hits": "",
        }

    anchor: Mapping[str, float] | None = None
    strategy = "random_exploration"
    if average_winner_weights and attempt % 3 == 0:
        anchor = average_winner_weights
        strategy = "winner_average_mutation"
    elif best_current_weights and attempt % 5 == 0:
        anchor = best_current_weights
        strategy = "best_current_mutation"

    values: Dict[str, float] = {}
    if anchor:
        for component in WEIGHT_COMPONENTS:
            base = max(0.0001, float(anchor.get(component, DEFAULT_EXHAUSTIVE_WEIGHTS[component])))
            jitter = rng.lognormvariate(0.0, 0.65)
            values[component] = base * jitter
            if rng.random() < 0.05:
                values[component] = 0.0
    else:
        for component in WEIGHT_COMPONENTS:
            base = float(DEFAULT_EXHAUSTIVE_WEIGHTS[component])
            if rng.random() < 0.10:
                values[component] = 0.0
            else:
                values[component] = rng.gammavariate(0.7 + base * 14.0, 1.0)
    return resolve_exhaustive_weights(values), {
        "weight_strategy": strategy,
        "elite_source_attempts": "",
        "elite_source_hits": "",
    }


def _evaluate_attempt(
    *,
    train_df: pd.DataFrame,
    target_concurso: int,
    actual_nums: Sequence[int],
    attempt_history: pd.DataFrame,
    climate_features: pd.DataFrame | None,
    weights: Mapping[str, float],
    top_games: int,
    exhaustive_limit: int | None,
    max_overlap: int,
    draw_hour: int,
    draw_minute: int,
    cache_dir: Path,
) -> Dict[str, object]:
    cache_build = _build_component_cache(
        train_df=train_df,
        target_concurso=target_concurso,
        climate_features=climate_features,
        exhaustive_limit=exhaustive_limit,
        draw_hour=draw_hour,
        draw_minute=draw_minute,
        cache_dir=cache_dir,
    )
    candidates, cache_rank = _load_cached_top_candidates(
        target_concurso=target_concurso,
        cache_dir=cache_dir,
        weights=weights,
        top_games=max(2, int(top_games)),
    )
    if candidates.empty:
        raise ValueError("Motor exaustivo nao gerou candidatos para a tentativa.")

    candidates = _apply_calibration_novelty(
        candidates,
        attempts=attempt_history,
        target_concurso=target_concurso,
    )
    final_games = select_final_games(candidates, max_overlap=max_overlap)
    game_1 = str(final_games.iloc[0]["nums"])
    game_2 = str(final_games.iloc[1]["nums"])
    hits_1 = compute_hits(_parse_nums(game_1), actual_nums)
    hits_2 = compute_hits(_parse_nums(game_2), actual_nums)
    best_hits = max(hits_1, hits_2)
    best_game = game_1 if hits_1 >= hits_2 else game_2
    return {
        "jogo_1": game_1,
        "acertos_jogo_1": int(hits_1),
        "jogo_2": game_2,
        "acertos_jogo_2": int(hits_2),
        "melhor_jogo": best_game,
        "melhor_acerto": int(best_hits),
        "candidates_evaluated": int(candidates.iloc[0].get("total_combinacoes_avaliadas", 0)),
        "score_jogo_1": float(final_games.iloc[0].get("score_final", 0.0)),
        "score_jogo_2": float(final_games.iloc[1].get("score_final", 0.0)),
        "score_original_jogo_1": float(final_games.iloc[0].get("score_final_original", final_games.iloc[0].get("score_final", 0.0))),
        "score_original_jogo_2": float(final_games.iloc[1].get("score_final_original", final_games.iloc[1].get("score_final", 0.0))),
        "repeat_count_jogo_1": int(final_games.iloc[0].get("calibration_repeat_count", 0)),
        "repeat_count_jogo_2": int(final_games.iloc[1].get("calibration_repeat_count", 0)),
        "recent_overlap_jogo_1": int(final_games.iloc[0].get("calibration_recent_max_overlap", 0)),
        "recent_overlap_jogo_2": int(final_games.iloc[1].get("calibration_recent_max_overlap", 0)),
        "novelty_penalty_jogo_1": float(final_games.iloc[0].get("calibration_novelty_penalty", 0.0)),
        "novelty_penalty_jogo_2": float(final_games.iloc[1].get("calibration_novelty_penalty", 0.0)),
        "score_novelty_jogo_1": float(final_games.iloc[0].get("score_calibration_novelty", 100.0)),
        "score_novelty_jogo_2": float(final_games.iloc[1].get("score_calibration_novelty", 100.0)),
        "cache_status": str(cache_build["status"]),
        "cache_rows": int(cache_build["rows"]),
        "cache_path": str(cache_rank["cache_path"]),
    }


def load_calibration_lab_status(
    *,
    state_json_path: Path,
    attempts_csv_path: Path,
    winners_csv_path: Path,
    elites_csv_path: Path,
    average_weights_csv_path: Path,
    engine_weights_json_path: Path,
    recent_rows: int = 12,
) -> Dict[str, object]:
    state = _load_json(state_json_path)
    attempts = _read_csv(attempts_csv_path)
    winners = _read_csv(winners_csv_path)
    elites = _read_csv(elites_csv_path)
    average = _read_csv(average_weights_csv_path)
    engine_payload = _load_json(engine_weights_json_path)
    current_concurso = state.get("current_concurso")
    display_elites = elites
    if current_concurso not in (None, ""):
        try:
            display_elites = _elite_rows_for_target(elites, int(current_concurso)).head(int(recent_rows))
        except (TypeError, ValueError):
            display_elites = elites.tail(int(recent_rows))
    else:
        display_elites = elites.tail(int(recent_rows))
    return {
        "state": state,
        "recent_attempts": attempts.tail(int(recent_rows)).astype(object).where(pd.notna(attempts.tail(int(recent_rows))), None).to_dict(orient="records") if not attempts.empty else [],
        "winners": winners.tail(int(recent_rows)).astype(object).where(pd.notna(winners.tail(int(recent_rows))), None).to_dict(orient="records") if not winners.empty else [],
        "elites": display_elites.astype(object).where(pd.notna(display_elites), None).to_dict(orient="records") if not display_elites.empty else [],
        "average_weights": average.astype(object).where(pd.notna(average), None).to_dict(orient="records") if not average.empty else [],
        "engine_weights": engine_payload.get("weights", {}) if isinstance(engine_payload, dict) else {},
        "paths": {
            "state_json_path": str(state_json_path),
            "attempts_csv_path": str(attempts_csv_path),
            "winners_csv_path": str(winners_csv_path),
            "elites_csv_path": str(elites_csv_path),
            "average_weights_csv_path": str(average_weights_csv_path),
            "engine_weights_json_path": str(engine_weights_json_path),
        },
    }


def run_calibration_lab(
    concursos: pd.DataFrame,
    *,
    climate_features: pd.DataFrame | None,
    from_concurso: int,
    to_concurso: int | None,
    max_attempts: int,
    top_games: int,
    exhaustive_limit: int | None,
    max_overlap: int,
    seed: int,
    draw_hour: int,
    draw_minute: int,
    min_history: int,
    max_runtime_seconds: int,
    reset: bool,
    state_json_path: Path,
    attempts_csv_path: Path,
    winners_csv_path: Path,
    elites_csv_path: Path,
    summary_csv_path: Path,
    average_weights_csv_path: Path,
    excel_path: Path,
    engine_weights_json_path: Path,
    cache_dir: Path,
) -> CalibrationLabSummary:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
    if int(top_games) < 2:
        raise ValueError("--lab-top-games deve ser pelo menos 2.")
    if int(max_overlap) < 0 or int(max_overlap) > 15:
        raise ValueError("--max-overlap-final deve estar entre 0 e 15.")

    if reset:
        _delete_if_exists(
            [
                state_json_path,
                attempts_csv_path,
                winners_csv_path,
                elites_csv_path,
                summary_csv_path,
                average_weights_csv_path,
                excel_path,
            ]
        )

    started_perf = time.perf_counter()
    started_at = _now()
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce").astype("Int64")
    contests = [
        int(value)
        for value in df["concurso"].dropna().astype(int).tolist()
        if int(value) >= int(from_concurso) and (to_concurso is None or int(value) <= int(to_concurso))
    ]
    if not contests:
        raise ValueError("Nenhum concurso encontrado no intervalo solicitado para calibracao.")

    attempts_this_run = 0
    status = "running"
    state = _load_json(state_json_path)
    state.setdefault("created_at", started_at)
    state["started_last_run_at"] = started_at
    state["from_concurso"] = int(from_concurso)
    state["to_concurso"] = int(to_concurso) if to_concurso is not None else None
    state["top_games"] = int(top_games)
    state["exhaustive_limit"] = int(exhaustive_limit) if exhaustive_limit is not None else None
    state["max_overlap"] = int(max_overlap)
    state["draw_hour"] = int(draw_hour)
    state["draw_minute"] = int(draw_minute)
    state["cache_dir"] = str(cache_dir)
    state["status"] = status

    while True:
        attempts = _read_csv(attempts_csv_path)
        winners = _read_csv(winners_csv_path)
        elites = _sync_elites_from_attempts(attempts=attempts, elites_csv_path=elites_csv_path)
        average_winner_weights = _write_average_outputs(
            winners=winners,
            average_weights_csv_path=average_weights_csv_path,
            engine_weights_json_path=engine_weights_json_path,
        )
        solved = _solved_contests(winners)
        preferred = state.get("current_concurso")
        preferred_int = int(preferred) if preferred not in (None, "") else None
        target_concurso = _next_target(contests, solved, preferred_int)
        if target_concurso is None:
            status = "complete"
            state.update(
                {
                    "status": status,
                    "current_concurso": None,
                    "updated_at": _now(),
                    "message": "Todos os concursos do intervalo foram resolvidos com 15 pontos.",
                }
            )
            _write_json(state_json_path, state)
            _write_summary(summary_csv_path=summary_csv_path, state=state, attempts=attempts, winners=winners, elites=elites)
            break

        matches = df.index[df["concurso"].astype(int) == int(target_concurso)].tolist()
        if not matches:
            raise ValueError(f"Concurso {target_concurso} nao encontrado na base local.")
        target_idx = int(matches[0])
        if target_idx < int(min_history):
            solved.add(int(target_concurso))
            state["current_concurso"] = _next_target(contests, solved, None)
            state["updated_at"] = _now()
            _write_json(state_json_path, state)
            continue

        train_df = df.iloc[:target_idx].copy()
        target_row = df.iloc[target_idx]
        actual_nums = _nums_from_row(target_row)
        next_attempt = _last_attempt_for_target(attempts, target_concurso) + 1
        best_current = _best_for_target(attempts, target_concurso)
        target_elites = _elite_rows_for_target(elites, target_concurso)
        elite_stats = _elite_stats(elites, target_concurso)
        weights, weight_meta = _weights_for_attempt(
            target_concurso=target_concurso,
            attempt=next_attempt,
            seed=seed,
            average_winner_weights=average_winner_weights,
            best_current_weights=best_current.get("weights", {}),
            elite_rows=target_elites,
        )

        state.update(
            {
                "status": status,
                "current_concurso": int(target_concurso),
                "current_attempt": int(next_attempt),
                "best_hits_current": int(best_current.get("hits", 0)),
                "best_game_current": str(best_current.get("game", "")),
                "best_attempt_current": int(best_current.get("attempt", 0)),
                **elite_stats,
                "current_weight_strategy": str(weight_meta.get("weight_strategy", "")),
                "current_elite_source_attempts": str(weight_meta.get("elite_source_attempts", "")),
                "current_elite_source_hits": str(weight_meta.get("elite_source_hits", "")),
                "solved_contests": sorted(solved),
                "solved_count": int(len(solved)),
                "average_weights": average_winner_weights or {},
                "current_cache_status": "building_or_loading",
                "last_attempt_started_at": _now(),
                "updated_at": _now(),
            }
        )
        state.pop("current_actual_numbers", None)
        _write_json(state_json_path, state)

        attempt_started = time.perf_counter()
        evaluation = _evaluate_attempt(
            train_df=train_df,
            target_concurso=target_concurso,
            actual_nums=actual_nums,
            attempt_history=attempts,
            climate_features=climate_features,
            weights=weights,
            top_games=top_games,
            exhaustive_limit=exhaustive_limit,
            max_overlap=max_overlap,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            cache_dir=cache_dir,
        )
        elapsed_attempt = time.perf_counter() - attempt_started
        attempts_this_run += 1

        row: Dict[str, object] = {
            "target_concurso": int(target_concurso),
            "data_sorteio": str(target_row.get("data_sorteio", "")),
            "tentativa": int(next_attempt),
            "generated_at": _now(),
            "actual_numbers": _format_nums(actual_nums),
            "jogo_1": evaluation["jogo_1"],
            "acertos_jogo_1": int(evaluation["acertos_jogo_1"]),
            "jogo_2": evaluation["jogo_2"],
            "acertos_jogo_2": int(evaluation["acertos_jogo_2"]),
            "melhor_jogo": evaluation["melhor_jogo"],
            "melhor_acerto": int(evaluation["melhor_acerto"]),
            "encontrou_15": int(evaluation["melhor_acerto"] >= 15),
            "elapsed_seconds": round(float(elapsed_attempt), 6),
            "candidates_evaluated": int(evaluation["candidates_evaluated"]),
            "top_games": int(top_games),
            "exhaustive_limit": int(exhaustive_limit) if exhaustive_limit is not None else "",
            "max_overlap": int(max_overlap),
            "score_jogo_1": float(evaluation["score_jogo_1"]),
            "score_jogo_2": float(evaluation["score_jogo_2"]),
            "score_original_jogo_1": float(evaluation["score_original_jogo_1"]),
            "score_original_jogo_2": float(evaluation["score_original_jogo_2"]),
            "repeat_count_jogo_1": int(evaluation["repeat_count_jogo_1"]),
            "repeat_count_jogo_2": int(evaluation["repeat_count_jogo_2"]),
            "recent_overlap_jogo_1": int(evaluation["recent_overlap_jogo_1"]),
            "recent_overlap_jogo_2": int(evaluation["recent_overlap_jogo_2"]),
            "novelty_penalty_jogo_1": float(evaluation["novelty_penalty_jogo_1"]),
            "novelty_penalty_jogo_2": float(evaluation["novelty_penalty_jogo_2"]),
            "score_novelty_jogo_1": float(evaluation["score_novelty_jogo_1"]),
            "score_novelty_jogo_2": float(evaluation["score_novelty_jogo_2"]),
            "cache_status": str(evaluation["cache_status"]),
            "cache_rows": int(evaluation["cache_rows"]),
            "cache_path": str(evaluation["cache_path"]),
            "weight_strategy": str(weight_meta.get("weight_strategy", "")),
            "elite_source_attempts": str(weight_meta.get("elite_source_attempts", "")),
            "elite_source_hits": str(weight_meta.get("elite_source_hits", "")),
            "score_weights": format_exhaustive_weights(weights),
        }
        for component, value in weights.items():
            row[f"peso_{component}"] = round(float(value), 10)
        _append_csv(attempts_csv_path, row)

        attempts = _read_csv(attempts_csv_path)
        elites = _sync_elites_from_attempts(attempts=attempts, elites_csv_path=elites_csv_path)
        elite_stats = _elite_stats(elites, target_concurso)
        best_current = _best_for_target(attempts, target_concurso)
        state.update(
            {
                "last_attempt_finished_at": _now(),
                "last_attempt_elapsed_seconds": round(float(elapsed_attempt), 6),
                "last_hits_jogo_1": int(evaluation["acertos_jogo_1"]),
                "last_hits_jogo_2": int(evaluation["acertos_jogo_2"]),
                "last_best_hits": int(evaluation["melhor_acerto"]),
                "last_best_game": str(evaluation["melhor_jogo"]),
                "last_cache_status": str(evaluation["cache_status"]),
                "last_cache_rows": int(evaluation["cache_rows"]),
                "last_cache_path": str(evaluation["cache_path"]),
                "last_repeat_count_jogo_1": int(evaluation["repeat_count_jogo_1"]),
                "last_repeat_count_jogo_2": int(evaluation["repeat_count_jogo_2"]),
                "last_recent_overlap_jogo_1": int(evaluation["recent_overlap_jogo_1"]),
                "last_recent_overlap_jogo_2": int(evaluation["recent_overlap_jogo_2"]),
                "last_novelty_penalty_jogo_1": float(evaluation["novelty_penalty_jogo_1"]),
                "last_novelty_penalty_jogo_2": float(evaluation["novelty_penalty_jogo_2"]),
                "last_weight_strategy": str(weight_meta.get("weight_strategy", "")),
                "last_elite_source_attempts": str(weight_meta.get("elite_source_attempts", "")),
                "last_elite_source_hits": str(weight_meta.get("elite_source_hits", "")),
                "current_cache_status": str(evaluation["cache_status"]),
                "current_weight_strategy": str(weight_meta.get("weight_strategy", "")),
                **elite_stats,
                "best_hits_current": int(best_current.get("hits", 0)),
                "best_game_current": str(best_current.get("game", "")),
                "best_attempt_current": int(best_current.get("attempt", 0)),
                "total_attempts": int(len(attempts)),
                "attempts_this_run": int(attempts_this_run),
                "elapsed_seconds_current_run": round(float(time.perf_counter() - started_perf), 6),
                "updated_at": _now(),
            }
        )

        if int(evaluation["melhor_acerto"]) >= 15:
            winner_row = dict(row)
            winner_row["solved_at"] = _now()
            _append_csv(winners_csv_path, winner_row)
            winners = _read_csv(winners_csv_path)
            average_winner_weights = _write_average_outputs(
                winners=winners,
                average_weights_csv_path=average_weights_csv_path,
                engine_weights_json_path=engine_weights_json_path,
            )
            solved = _solved_contests(winners)
            state.update(
                {
                    "status": "running",
                    "last_solved_concurso": int(target_concurso),
                    "last_solved_attempt": int(next_attempt),
                    "last_solved_at": _now(),
                    "solved_contests": sorted(solved),
                    "solved_count": int(len(solved)),
                    "average_weights": average_winner_weights or {},
                    "current_concurso": _next_target(contests, solved, None),
                    "current_attempt": 0,
                    "best_hits_current": 0,
                    "best_game_current": "",
                    "best_attempt_current": 0,
                    "updated_at": _now(),
                }
            )

        winners = _read_csv(winners_csv_path)
        _write_json(state_json_path, state)
        _write_summary(summary_csv_path=summary_csv_path, state=state, attempts=attempts, winners=winners, elites=elites)

        if attempts_this_run % 20 == 0 or int(evaluation["melhor_acerto"]) >= 15:
            _write_excel_snapshot(
                excel_path=excel_path,
                attempts=attempts,
                winners=winners,
                elites=elites,
                average_weights_csv_path=average_weights_csv_path,
                summary_csv_path=summary_csv_path,
            )

        if int(max_attempts) > 0 and attempts_this_run >= int(max_attempts):
            status = "paused_by_attempt_limit"
            state["status"] = status
            state["updated_at"] = _now()
            _write_json(state_json_path, state)
            break
        if int(max_runtime_seconds) > 0 and (time.perf_counter() - started_perf) >= int(max_runtime_seconds):
            status = "paused_by_runtime_limit"
            state["status"] = status
            state["updated_at"] = _now()
            _write_json(state_json_path, state)
            break

    attempts = _read_csv(attempts_csv_path)
    winners = _read_csv(winners_csv_path)
    elites = _sync_elites_from_attempts(attempts=attempts, elites_csv_path=elites_csv_path)
    best_current = _best_for_target(attempts, int(state["current_concurso"])) if state.get("current_concurso") else {"hits": 0, "game": ""}
    _write_average_outputs(
        winners=winners,
        average_weights_csv_path=average_weights_csv_path,
        engine_weights_json_path=engine_weights_json_path,
    )
    _write_summary(summary_csv_path=summary_csv_path, state=state, attempts=attempts, winners=winners, elites=elites)
    _write_excel_snapshot(
        excel_path=excel_path,
        attempts=attempts,
        winners=winners,
        elites=elites,
        average_weights_csv_path=average_weights_csv_path,
        summary_csv_path=summary_csv_path,
    )

    return CalibrationLabSummary(
        status=str(state.get("status", status)),
        current_concurso=int(state["current_concurso"]) if state.get("current_concurso") else None,
        attempts_this_run=int(attempts_this_run),
        total_attempts=int(len(attempts)),
        solved_contests=int(len(_solved_contests(winners))),
        best_hits_current=int(best_current.get("hits", 0)),
        best_game_current=str(best_current.get("game", "")),
        elapsed_seconds=round(float(time.perf_counter() - started_perf), 6),
        attempts_csv_path=str(attempts_csv_path),
        winners_csv_path=str(winners_csv_path),
        state_json_path=str(state_json_path),
        average_weights_csv_path=str(average_weights_csv_path),
        engine_weights_json_path=str(engine_weights_json_path),
    )
