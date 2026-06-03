from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from .context_features import (
    BRASILIA_TZ,
    CLIMATE_CONTEXT_COLUMNS,
    digital_root,
    estacao_do_ano,
    moon_phase,
    normalize_context_text,
)
from .exhaustive_optimizer import (
    DEFAULT_EXHAUSTIVE_WEIGHTS,
    NUMBERS,
    resolve_exhaustive_weights,
)
from .normalize import DEZENAS
from .optimizer import (
    _delays,
    _pair_counter,
)
from .temporal_deep import temporal_deep_number_scores


COMPONENTS = list(DEFAULT_EXHAUSTIVE_WEIGHTS.keys())


@dataclass(frozen=True)
class EngineCalibrationSummary:
    rows: int
    contests: int
    first_concurso: int
    last_concurso: int
    weights_json_path: str
    results_csv_path: str
    summary_csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Calibracao do Motor",
                f"Concursos calibrados: {self.contests}",
                f"Primeiro concurso: {self.first_concurso}",
                f"Ultimo concurso: {self.last_concurso}",
                f"Linhas: {self.rows}",
                f"Pesos calibrados: {self.weights_json_path}",
                f"CSV resultados: {self.results_csv_path}",
                f"CSV resumo: {self.summary_csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Pesos calibrados por walk-forward, usando apenas historico anterior a cada concurso.",
            ]
        )


def nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def _score_0_100(penalty: float) -> float:
    return round(max(0.0, min(100.0, 100.0 - float(penalty))), 6)


def _frequency_number_scores(counter, sample_size: int, *, shrink_base: float = 50.0) -> Dict[int, float]:
    if sample_size <= 0:
        return {n: 50.0 for n in NUMBERS}
    expected = sample_size * 15.0 / 25.0
    shrink = min(1.0, sample_size / shrink_base)
    scores: Dict[int, float] = {}
    for n in NUMBERS:
        raw = 50.0 + (float(counter.get(n, 0)) - expected) * 7.0
        scores[n] = round(max(0.0, min(100.0, 50.0 + (raw - 50.0) * shrink)), 6)
    return scores


def _delay_number_scores(delays: Mapping[int, int]) -> Dict[int, float]:
    values = list(delays.values())
    median_delay = float(pd.Series(values).median()) if values else 0.0
    return {
        n: _score_0_100(abs(float(delays.get(n, 0)) - median_delay) * 1.4)
        for n in NUMBERS
    }


def _pair_centrality_scores(draws: Sequence[Sequence[int]]) -> Dict[int, float]:
    recent_draws = list(draws[-100:])
    if not recent_draws:
        return {n: 50.0 for n in NUMBERS}
    pair_freq = _pair_counter(recent_draws)
    raw = {
        n: sum(float(pair_freq.get(tuple(sorted((n, other))), 0.0)) for other in NUMBERS if other != n)
        for n in NUMBERS
    }
    values = pd.Series(list(raw.values()), dtype="float64")
    low = float(values.quantile(0.10))
    high = float(values.quantile(0.90))
    span = max(1.0, high - low)
    return {n: round(max(0.0, min(100.0, (raw[n] - low) / span * 100.0)), 6) for n in NUMBERS}


def _fixed_number_scores(values: Mapping[int, float], default: float = 50.0) -> Dict[int, float]:
    return {n: float(values.get(n, default)) for n in NUMBERS}


def _delta_drawn_vs_not(scores: Mapping[int, float], actual: Sequence[int]) -> Tuple[float, float, float]:
    actual_set = set(int(n) for n in actual)
    drawn = [float(scores.get(n, 50.0)) for n in NUMBERS if n in actual_set]
    not_drawn = [float(scores.get(n, 50.0)) for n in NUMBERS if n not in actual_set]
    actual_mean = float(pd.Series(drawn).mean()) if drawn else 50.0
    baseline_mean = float(pd.Series(not_drawn).mean()) if not_drawn else 50.0
    return actual_mean, baseline_mean, actual_mean - baseline_mean


def _context_keys_for_row(row: pd.Series, *, draw_hour: int, draw_minute: int) -> List[str]:
    draw_date = pd.to_datetime(row["data_sorteio"], errors="coerce")
    if pd.isna(draw_date):
        return []
    draw_datetime = datetime.combine(draw_date.date(), time(hour=int(draw_hour), minute=int(draw_minute)), tzinfo=BRASILIA_TZ)
    moon = moon_phase(draw_datetime)
    keys = [
        f"weekday:{int(draw_date.isoweekday())}",
        f"month:{int(draw_date.month)}",
        f"bimester:{int((draw_date.month - 1) // 2 + 1)}",
        f"quarter:{int((draw_date.month - 1) // 3 + 1)}",
        f"semester:{int((draw_date.month - 1) // 6 + 1)}",
        f"season:{estacao_do_ano(draw_date)}",
        f"moon:{moon['fase_lua']}",
        f"numerology_date:{digital_root(int(draw_date.strftime('%Y%m%d')))}",
        f"numerology_concurso:{digital_root(int(row['concurso']))}",
        f"numerology_day_month:{digital_root(int(draw_date.day) + int(draw_date.month))}",
    ]
    local = normalize_context_text(row.get("local_sorteio", ""))
    cidade = normalize_context_text(row.get("cidade_sorteio", ""))
    uf = normalize_context_text(row.get("uf_sorteio", ""))
    bairro = normalize_context_text(row.get("bairro_sorteio", "")) if "bairro_sorteio" in row.index else ""
    if local:
        keys.append(f"local:{local}")
    if cidade:
        keys.append(f"cidade:{cidade}")
    if uf:
        keys.append(f"uf:{uf}")
    if bairro:
        keys.append(f"bairro:{bairro}")
    if cidade and uf:
        keys.append(f"cidade_uf:{cidade}|{uf}")
    return keys


def _climate_keys_for_mapping(climate: Mapping[str, object] | None) -> List[str]:
    if not climate:
        return []
    keys: List[str] = []
    for column, prefix in CLIMATE_CONTEXT_COLUMNS:
        value = climate.get(column)
        if value is None or pd.isna(value) or str(value) == "indisponivel":
            continue
        keys.append(f"{prefix}:{value}")
    return keys


def _weighted_key_scores(
    *,
    keys: Sequence[str],
    counters: Mapping[str, Counter[int]],
    samples: Mapping[str, int],
    default: float = 50.0,
) -> Dict[int, float]:
    if not keys:
        return {n: default for n in NUMBERS}
    out = {n: 0.0 for n in NUMBERS}
    total = 0
    for key in keys:
        sample_size = int(samples.get(key, 0))
        score_map = _frequency_number_scores(counters.get(key, Counter()), sample_size)
        for n in NUMBERS:
            out[n] += score_map[n]
        total += 1
    if total <= 0:
        return {n: default for n in NUMBERS}
    return {n: round(out[n] / total, 6) for n in NUMBERS}


def _temporal_incremental_scores(
    *,
    row: pd.Series,
    temporal_counters: Mapping[str, Counter[int]],
    temporal_samples: Mapping[str, int],
    previous_draws: Sequence[Tuple[pd.Timestamp, Sequence[int]]],
) -> Dict[int, float]:
    draw_date = pd.to_datetime(row["data_sorteio"], errors="coerce")
    if pd.isna(draw_date):
        return {n: 50.0 for n in NUMBERS}
    weekday = int(draw_date.isoweekday())
    bimester = int((draw_date.month - 1) // 2 + 1)
    quarter = int((draw_date.month - 1) // 3 + 1)
    semester = int((draw_date.month - 1) // 6 + 1)
    recent_15 = Counter()
    recent_30 = Counter()
    sample_15 = sample_30 = 0
    for prev_date, nums in previous_draws:
        if draw_date - pd.Timedelta(days=30) <= prev_date < draw_date:
            recent_30.update(nums)
            sample_30 += 1
            if draw_date - pd.Timedelta(days=15) <= prev_date < draw_date:
                recent_15.update(nums)
                sample_15 += 1
    maps = {
        "weekday": _frequency_number_scores(temporal_counters.get(f"weekday:{weekday}", Counter()), temporal_samples.get(f"weekday:{weekday}", 0)),
        "recent_15d": _frequency_number_scores(recent_15, sample_15),
        "recent_30d": _frequency_number_scores(recent_30, sample_30),
        "bimestre": _frequency_number_scores(temporal_counters.get(f"bimester:{bimester}", Counter()), temporal_samples.get(f"bimester:{bimester}", 0)),
        "trimestre": _frequency_number_scores(temporal_counters.get(f"quarter:{quarter}", Counter()), temporal_samples.get(f"quarter:{quarter}", 0)),
        "semestre": _frequency_number_scores(temporal_counters.get(f"semester:{semester}", Counter()), temporal_samples.get(f"semester:{semester}", 0)),
    }
    weights = {"weekday": 0.22, "recent_15d": 0.18, "recent_30d": 0.16, "bimestre": 0.18, "trimestre": 0.13, "semestre": 0.13}
    return {n: round(sum(weights[key] * maps[key][n] for key in weights), 6) for n in NUMBERS}


def run_engine_calibration(
    concursos: pd.DataFrame,
    *,
    climate_features: pd.DataFrame | None,
    from_concurso: int = 2500,
    to_concurso: int | None = None,
    baseline_samples: int = 30,
    seed: int = 123,
    draw_hour: int = 20,
    draw_minute: int = 0,
    weights_json_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado.")
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    max_target = int(to_concurso) if to_concurso else int(df["concurso"].max())
    climate_by_concurso: Dict[int, Mapping[str, object]] = {}
    if climate_features is not None and not climate_features.empty and "concurso" in climate_features.columns:
        climate_df = climate_features.copy()
        climate_df["concurso"] = pd.to_numeric(climate_df["concurso"], errors="coerce")
        climate_by_concurso = {
            int(row["concurso"]): row.to_dict()
            for _, row in climate_df.dropna(subset=["concurso"]).iterrows()
        }

    context_counters: Dict[str, Counter[int]] = defaultdict(Counter)
    context_samples: Counter[str] = Counter()
    climate_counters: Dict[str, Counter[int]] = defaultdict(Counter)
    climate_samples: Counter[str] = Counter()
    temporal_counters: Dict[str, Counter[int]] = defaultdict(Counter)
    temporal_samples: Counter[str] = Counter()
    previous_draws: deque[Tuple[pd.Timestamp, Sequence[int]]] = deque()
    recent_100: deque[Sequence[int]] = deque()
    recent_freq: Counter[int] = Counter()
    pair_recent: Counter[Tuple[int, int]] = Counter()
    last_seen = {n: None for n in NUMBERS}
    last_draw: set[int] = set()
    keep_hits: Counter[int] = Counter()
    keep_opp: Counter[int] = Counter()
    enter_hits: Counter[int] = Counter()
    enter_opp: Counter[int] = Counter()
    rows: List[Dict[str, object]] = []

    for idx, row in df.iterrows():
        concurso = int(row["concurso"])
        draw_date = pd.to_datetime(row["data_sorteio"], errors="coerce")
        nums = nums_from_row(row)
        if concurso >= int(from_concurso) and concurso <= max_target and idx >= 10:
            delays = {n: idx if last_seen[n] is None else idx - int(last_seen[n]) - 1 for n in NUMBERS}
            context_scores = _weighted_key_scores(
                keys=_context_keys_for_row(row, draw_hour=draw_hour, draw_minute=draw_minute),
                counters=context_counters,
                samples=context_samples,
            )
            target_roots = {
                digital_root(int(pd.Timestamp(draw_date).strftime("%Y%m%d"))) if not pd.isna(draw_date) else 0,
                digital_root(concurso),
            }
            for n in NUMBERS:
                context_scores[n] = round(max(0.0, min(100.0, context_scores[n] + (5.0 if digital_root(n) in target_roots else -1.5))), 6)
            climate_scores = _weighted_key_scores(
                keys=_climate_keys_for_mapping(climate_by_concurso.get(concurso)),
                counters=climate_counters,
                samples=climate_samples,
            )
            temporal_scores = _temporal_incremental_scores(
                row=row,
                temporal_counters=temporal_counters,
                temporal_samples=temporal_samples,
                previous_draws=list(previous_draws),
            )
            transition_scores = {}
            for n in NUMBERS:
                if not last_draw:
                    transition_scores[n] = 50.0
                elif n in last_draw:
                    transition_scores[n] = round((keep_hits[n] + 1.0) / (keep_opp[n] + 2.0) * 100.0, 6)
                else:
                    transition_scores[n] = round((enter_hits[n] + 1.0) / (enter_opp[n] + 2.0) * 100.0, 6)
            component_scores = {
                "estatistico": _fixed_number_scores({}, 50.0),
                "historico": _frequency_number_scores(recent_freq, len(recent_100)),
                "atraso": _delay_number_scores(delays),
                "combinatorio": _pair_centrality_scores(list(recent_100)),
                "localidade_numerologia": context_scores,
                "climatico": climate_scores,
                "temporal_profundo": temporal_scores,
                "cenarios": _fixed_number_scores({1: 63.0, 5: 60.0, 13: 62.0, 21: 59.0, 22: 61.0, 25: 58.0}, 50.0),
                "contrarian": _fixed_number_scores({1: 70.0, 13: 68.0, 22: 66.0, 5: 58.0, 21: 58.0, 25: 58.0}, 50.0),
                "transicao": transition_scores,
                "nao_repeticao_exata": _fixed_number_scores({}, 50.0),
            }
            for component in COMPONENTS:
                actual_mean, baseline_mean, delta = _delta_drawn_vs_not(component_scores[component], nums)
                rows.append(
                    {
                        "concurso": concurso,
                        "data_sorteio": pd.Timestamp(draw_date).date().isoformat() if not pd.isna(draw_date) else "",
                        "componente": component,
                        "score_jogo_real": round(actual_mean, 6),
                        "score_baseline_aleatorio_medio": round(baseline_mean, 6),
                        "delta_real_vs_baseline": round(delta, 6),
                        "baseline_samples": int(baseline_samples),
                        "metodo_calibracao": "incremental_dezena_walk_forward_v3",
                    }
                )

        if not pd.isna(draw_date):
            context_keys = _context_keys_for_row(row, draw_hour=draw_hour, draw_minute=draw_minute)
            climate_keys = _climate_keys_for_mapping(climate_by_concurso.get(concurso))
            for key in context_keys:
                context_counters[key].update(nums)
                context_samples[key] += 1
            for key in climate_keys:
                climate_counters[key].update(nums)
                climate_samples[key] += 1
            weekday = int(draw_date.isoweekday())
            bimester = int((draw_date.month - 1) // 2 + 1)
            quarter = int((draw_date.month - 1) // 3 + 1)
            semester = int((draw_date.month - 1) // 6 + 1)
            for key in [f"weekday:{weekday}", f"bimester:{bimester}", f"quarter:{quarter}", f"semester:{semester}"]:
                temporal_counters[key].update(nums)
                temporal_samples[key] += 1
            previous_draws.append((pd.Timestamp(draw_date), nums))
        if last_draw:
            current_set = set(nums)
            for n in NUMBERS:
                if n in last_draw:
                    keep_opp[n] += 1
                    keep_hits[n] += int(n in current_set)
                else:
                    enter_opp[n] += 1
                    enter_hits[n] += int(n in current_set)
        for n in nums:
            last_seen[int(n)] = idx
        last_draw = set(nums)
        recent_100.append(nums)
        recent_freq.update(nums)
        pair_recent.update(tuple(pair) for pair in combinations(nums, 2))
        while len(recent_100) > 100:
            old = recent_100.popleft()
            recent_freq.subtract(old)
            for n in list(recent_freq):
                if recent_freq[n] <= 0:
                    del recent_freq[n]
            pair_recent.subtract(tuple(pair) for pair in combinations(old, 2))
            for pair in list(pair_recent):
                if pair_recent[pair] <= 0:
                    del pair_recent[pair]

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("Calibracao nao gerou resultados.")
    summary_rows: List[Dict[str, object]] = []
    positive: Dict[str, float] = {}
    for component in COMPONENTS:
        group = results[results["componente"] == component]
        avg_actual = float(group["score_jogo_real"].mean())
        avg_baseline = float(group["score_baseline_aleatorio_medio"].mean())
        avg_delta = float(group["delta_real_vs_baseline"].mean())
        positive[component] = max(0.0, avg_delta)
        summary_rows.append(
            {
                "componente": component,
                "concursos": int(group["concurso"].nunique()),
                "score_real_medio": round(avg_actual, 6),
                "score_baseline_medio": round(avg_baseline, 6),
                "delta_medio": round(avg_delta, 6),
                "status_calibracao": "ativo" if avg_delta > 0 else "stand_by",
            }
        )
    calibrated = dict(DEFAULT_EXHAUSTIVE_WEIGHTS) if sum(positive.values()) <= 0 else resolve_exhaustive_weights(positive)
    summary = pd.DataFrame(summary_rows).sort_values("delta_medio", ascending=False).reset_index(drop=True)
    summary["peso_calibrado"] = summary["componente"].map(calibrated).fillna(0.0)
    payload = {
        "model": "engine_calibration_walk_forward_v3",
        "from_concurso": int(from_concurso),
        "to_concurso": int(max_target),
        "baseline_samples": int(baseline_samples),
        "draw_hour": int(draw_hour),
        "draw_minute": int(draw_minute),
        "weights": calibrated,
        "stand_by_components": summary.loc[summary["status_calibracao"] == "stand_by", "componente"].tolist(),
    }
    weights_json_path.parent.mkdir(parents=True, exist_ok=True)
    weights_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return results, summary, payload
