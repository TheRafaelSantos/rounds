from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from math import ceil
from pathlib import Path
from statistics import median
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from .context_features import build_target_context
from .exhaustive_optimizer import (
    EXHAUSTIVE_SOURCE_MODEL,
    TOTAL_COMBINATIONS,
    build_exhaustive_candidates,
    format_exhaustive_weights,
    resolve_exhaustive_weights,
)
from .mandel_strategy import greedy_reduced_closure
from .normalize import DEZENAS
from .selection_guard import enrich_candidates_with_decision_guard
from .storage import sanitize_dataframe_for_tabular_output
from .supervised_calibration import _apply_weights, _score_candidate_table
from .top50_refinement import apply_top50_refinement


NUMBERS = tuple(range(1, 26))
GRID_ROWS = tuple(range(5))
GRID_COLS = tuple(range(5))
BORDER = {1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25}
CENTER_3X3 = {7, 8, 9, 12, 13, 14, 17, 18, 19}
DIAGONAL_1 = {1, 7, 13, 19, 25}
DIAGONAL_2 = {5, 9, 13, 17, 21}
TOP100_SOURCE_MODEL = "top100_ranker_v1_hard_negatives_advanced_studies"


@dataclass(frozen=True)
class Top100Summary:
    concurso_alvo: int
    generated_at: str
    top_count: int
    top_pool: int
    selected_rows: int
    data_proximo_concurso: str
    prediction_csv_path: str
    report_path: str
    excel_path: str
    metodo: str = TOP100_SOURCE_MODEL

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Motor Top 100 / Top 50",
                f"Concurso-alvo: {self.concurso_alvo}",
                f"Gerado em: {self.generated_at}",
                f"Data do proximo concurso: {self.data_proximo_concurso}",
                f"Jogos selecionados: {self.selected_rows}",
                f"Top solicitado: {self.top_count}",
                f"Pool analisado: {self.top_pool}",
                f"Metodo: {self.metodo}",
                f"CSV: {self.prediction_csv_path}",
                f"Relatorio: {self.report_path}",
                f"Excel: {self.excel_path}",
                "Aviso: ranking estatistico; nao existe garantia de acerto em sorteios aleatorios.",
            ]
        )


@dataclass(frozen=True)
class Top100BacktestSummary:
    concursos_avaliados: int
    top_count: int
    top_pool: int
    hit_top10: int
    hit_top50: int
    hit_top100: int
    taxa_top10: float
    taxa_top50: float
    taxa_top100: float
    rank_diagnostico_medio: float
    results_csv_path: str
    summary_csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Backtest Top 100 / Top 50",
                f"Concursos avaliados: {self.concursos_avaliados}",
                f"Top solicitado: {self.top_count}",
                f"Pool por concurso: {self.top_pool}",
                f"Hits Top 10: {self.hit_top10} ({self.taxa_top10:.2f}%)",
                f"Hits Top 50: {self.hit_top50} ({self.taxa_top50:.2f}%)",
                f"Hits Top 100: {self.hit_top100} ({self.taxa_top100:.2f}%)",
                f"Rank diagnostico medio: {self.rank_diagnostico_medio:.2f}",
                f"CSV resultados: {self.results_csv_path}",
                f"CSV resumo: {self.summary_csv_path}",
                f"Excel: {self.excel_path}",
            ]
        )


@dataclass
class AdvancedStudyModel:
    draws: List[Tuple[int, ...]]
    pair_freq: Counter[Tuple[int, int]]
    trio_freq: Counter[Tuple[int, int, int]]
    quartet_freq: Counter[Tuple[int, int, int, int]]
    pair_last_seen: Dict[Tuple[int, int], int]
    trio_last_seen: Dict[Tuple[int, int, int], int]
    quartet_last_seen: Dict[Tuple[int, int, int, int], int]
    omit_number_freq: Counter[int]
    omit_pair_freq: Counter[Tuple[int, int]]
    number_regime_scores: Dict[int, float]
    graph_centrality: Dict[int, float]
    medians: Dict[str, object]
    expected: Dict[str, float]
    n_draws: int


def _nums_from_row(row: pd.Series) -> Tuple[int, ...]:
    return tuple(sorted(int(row[col]) for col in DEZENAS))


def _parse_nums(text: str) -> Tuple[int, ...]:
    nums = tuple(sorted(int(part) for part in str(text).split()))
    if len(nums) != 15 or len(set(nums)) != 15 or any(n < 1 or n > 25 for n in nums):
        raise ValueError(f"Candidato invalido: {text}")
    return nums


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _expanded_top_pool(top_count: int, requested_top_pool: int) -> int:
    return int(requested_top_pool)


def _score_closeness(value: float, target: float, scale: float) -> float:
    scale = max(0.000001, float(scale))
    penalty = abs(float(value) - float(target)) / scale * 18.0
    return round(max(0.0, min(100.0, 100.0 - penalty)), 6)


def _avg(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _grid_counts(nums: Sequence[int]) -> Tuple[List[int], List[int]]:
    rows = [0, 0, 0, 0, 0]
    cols = [0, 0, 0, 0, 0]
    for n in nums:
        value = int(n) - 1
        rows[value // 5] += 1
        cols[value % 5] += 1
    return rows, cols


def _mod_counts(nums: Sequence[int], modulo: int) -> List[int]:
    counts = [0 for _ in range(int(modulo))]
    for n in nums:
        counts[int(n) % int(modulo)] += 1
    return counts


def _distribution_score(values: Sequence[int], targets: Sequence[float], *, scale: float = 1.4) -> float:
    penalty = sum(abs(float(v) - float(t)) for v, t in zip(values, targets))
    return round(max(0.0, min(100.0, 100.0 - penalty * float(scale) * 8.0)), 6)


def _build_counter_and_last_seen(draws: Sequence[Sequence[int]], size: int) -> Tuple[Counter[Tuple[int, ...]], Dict[Tuple[int, ...], int]]:
    counter: Counter[Tuple[int, ...]] = Counter()
    last_seen: Dict[Tuple[int, ...], int] = {}
    for idx, draw in enumerate(draws, start=1):
        for combo in combinations(tuple(sorted(draw)), int(size)):
            counter[combo] += 1
            last_seen[combo] = idx
    return counter, last_seen


def _build_advanced_model(concursos: pd.DataFrame) -> AdvancedStudyModel:
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    draws = [_nums_from_row(row) for _, row in df.iterrows()]
    n_draws = len(draws)
    pair_freq, pair_last = _build_counter_and_last_seen(draws, 2)
    trio_freq, trio_last = _build_counter_and_last_seen(draws, 3)
    quartet_freq, quartet_last = _build_counter_and_last_seen(draws, 4)
    omit_number_freq: Counter[int] = Counter()
    omit_pair_freq: Counter[Tuple[int, int]] = Counter()

    row_profiles: List[List[int]] = []
    col_profiles: List[List[int]] = []
    mod3_profiles: List[List[int]] = []
    mod4_profiles: List[List[int]] = []
    mod5_profiles: List[List[int]] = []
    border_counts: List[int] = []
    center_counts: List[int] = []
    diag_counts: List[int] = []

    for draw in draws:
        selected = set(draw)
        omitted = tuple(n for n in NUMBERS if n not in selected)
        omit_number_freq.update(omitted)
        omit_pair_freq.update(tuple(pair) for pair in combinations(omitted, 2))
        rows, cols = _grid_counts(draw)
        row_profiles.append(rows)
        col_profiles.append(cols)
        mod3_profiles.append(_mod_counts(draw, 3))
        mod4_profiles.append(_mod_counts(draw, 4))
        mod5_profiles.append(_mod_counts(draw, 5))
        border_counts.append(len(selected & BORDER))
        center_counts.append(len(selected & CENTER_3X3))
        diag_counts.append(max(len(selected & DIAGONAL_1), len(selected & DIAGONAL_2)))

    def median_profile(profiles: Sequence[Sequence[int]], length: int) -> List[float]:
        if not profiles:
            return [15.0 / length for _ in range(length)]
        return [float(median([profile[idx] for profile in profiles])) for idx in range(length)]

    expected_pair = n_draws * 105.0 / 300.0
    expected_trio = n_draws * 455.0 / 2300.0
    expected_quartet = n_draws * 1365.0 / 12650.0
    expected_omit_pair = n_draws * 45.0 / 300.0
    graph_centrality = {
        n: float(sum(pair_freq.get(tuple(sorted((n, other))), 0) for other in NUMBERS if other != n))
        for n in NUMBERS
    }

    number_regime_scores: Dict[int, float] = {}
    for n in NUMBERS:
        weighted = 0.0
        total_weight = 0.0
        for window, weight in [(30, 0.32), (100, 0.28), (500, 0.22), (n_draws, 0.18)]:
            if n_draws <= 0:
                continue
            recent_draws = draws[-min(int(window), n_draws) :]
            sample = len(recent_draws)
            count = sum(1 for draw in recent_draws if n in set(draw))
            expected_count = sample * 15.0 / 25.0
            score = _score_closeness(count, expected_count, max(1.0, expected_count * 0.20))
            weighted += score * weight
            total_weight += weight
        number_regime_scores[n] = round(weighted / total_weight, 6) if total_weight else 50.0

    medians: Dict[str, object] = {
        "rows": median_profile(row_profiles, 5),
        "cols": median_profile(col_profiles, 5),
        "mod3": median_profile(mod3_profiles, 3),
        "mod4": median_profile(mod4_profiles, 4),
        "mod5": median_profile(mod5_profiles, 5),
        "border": float(median(border_counts)) if border_counts else 9.0,
        "center": float(median(center_counts)) if center_counts else 6.0,
        "diag": float(median(diag_counts)) if diag_counts else 3.0,
    }
    expected = {
        "pair_freq": expected_pair,
        "trio_freq": expected_trio,
        "quartet_freq": expected_quartet,
        "omit_number_freq": n_draws * 10.0 / 25.0,
        "omit_pair_freq": expected_omit_pair,
        "pair_delay": max(1.0, n_draws / max(1.0, expected_pair)),
        "trio_delay": max(1.0, n_draws / max(1.0, expected_trio)),
        "quartet_delay": max(1.0, n_draws / max(1.0, expected_quartet)),
        "graph_centrality": _avg(list(graph_centrality.values())),
    }
    return AdvancedStudyModel(
        draws=draws,
        pair_freq=pair_freq,
        trio_freq=trio_freq,
        quartet_freq=quartet_freq,
        pair_last_seen=pair_last,
        trio_last_seen=trio_last,
        quartet_last_seen=quartet_last,
        omit_number_freq=omit_number_freq,
        omit_pair_freq=omit_pair_freq,
        number_regime_scores=number_regime_scores,
        graph_centrality=graph_centrality,
        medians=medians,
        expected=expected,
        n_draws=n_draws,
    )


def _family_combo_score(
    nums: Sequence[int],
    *,
    size: int,
    freq: Counter[Tuple[int, ...]],
    last_seen: Mapping[Tuple[int, ...], int],
    n_draws: int,
    expected_freq: float,
    expected_delay: float,
) -> float:
    values = []
    delays = []
    for combo in combinations(tuple(sorted(nums)), int(size)):
        values.append(float(freq.get(combo, 0)))
        delays.append(float(n_draws + 1 - int(last_seen.get(combo, 0))))
    avg_freq = _avg(values)
    avg_delay = _avg(delays)
    freq_score = _score_closeness(avg_freq, expected_freq, max(1.0, expected_freq * 0.20))
    delay_score = _score_closeness(avg_delay, expected_delay, max(1.0, expected_delay * 0.55))
    return round((freq_score * 0.62) + (delay_score * 0.38), 6)


def _score_advanced_studies(nums: Sequence[int], row: Mapping[str, object], model: AdvancedStudyModel) -> Dict[str, object]:
    selected = tuple(sorted(int(n) for n in nums))
    selected_set = set(selected)
    omitted = tuple(n for n in NUMBERS if n not in selected_set)

    pair_score = _family_combo_score(
        selected,
        size=2,
        freq=model.pair_freq,
        last_seen=model.pair_last_seen,
        n_draws=model.n_draws,
        expected_freq=model.expected["pair_freq"],
        expected_delay=model.expected["pair_delay"],
    )
    trio_score = _family_combo_score(
        selected,
        size=3,
        freq=model.trio_freq,
        last_seen=model.trio_last_seen,
        n_draws=model.n_draws,
        expected_freq=model.expected["trio_freq"],
        expected_delay=model.expected["trio_delay"],
    )
    quartet_score = _family_combo_score(
        selected,
        size=4,
        freq=model.quartet_freq,
        last_seen=model.quartet_last_seen,
        n_draws=model.n_draws,
        expected_freq=model.expected["quartet_freq"],
        expected_delay=model.expected["quartet_delay"],
    )
    score_combinatorio_avancado = round((pair_score * 0.42) + (trio_score * 0.36) + (quartet_score * 0.22), 6)

    pair_values = [float(model.pair_freq.get(tuple(pair), 0)) for pair in combinations(selected, 2)]
    avg_pair_density = _avg(pair_values)
    selected_centrality = _avg([model.graph_centrality.get(n, 0.0) for n in selected])
    graph_density_score = _score_closeness(avg_pair_density, model.expected["pair_freq"], max(1.0, model.expected["pair_freq"] * 0.22))
    graph_centrality_score = _score_closeness(selected_centrality, model.expected["graph_centrality"], max(1.0, model.expected["graph_centrality"] * 0.12))
    score_grafo_dezenas = round((graph_density_score * 0.58) + (graph_centrality_score * 0.42), 6)

    omitted_number_avg = _avg([float(model.omit_number_freq.get(n, 0)) for n in omitted])
    omitted_pair_avg = _avg([float(model.omit_pair_freq.get(tuple(pair), 0)) for pair in combinations(omitted, 2)])
    complement_number_score = _score_closeness(omitted_number_avg, model.expected["omit_number_freq"], max(1.0, model.expected["omit_number_freq"] * 0.18))
    complement_pair_score = _score_closeness(omitted_pair_avg, model.expected["omit_pair_freq"], max(1.0, model.expected["omit_pair_freq"] * 0.24))
    score_complemento_ausentes = round((complement_number_score * 0.45) + (complement_pair_score * 0.55), 6)

    rows, cols = _grid_counts(selected)
    border_count = len(selected_set & BORDER)
    center_count = len(selected_set & CENTER_3X3)
    diag_count = max(len(selected_set & DIAGONAL_1), len(selected_set & DIAGONAL_2))
    score_rows = _distribution_score(rows, model.medians["rows"], scale=1.0)
    score_cols = _distribution_score(cols, model.medians["cols"], scale=1.0)
    score_border = _score_closeness(border_count, float(model.medians["border"]), 1.8)
    score_center = _score_closeness(center_count, float(model.medians["center"]), 1.8)
    score_diag = _score_closeness(diag_count, float(model.medians["diag"]), 1.4)
    score_geometria_volante = round((score_rows * 0.27) + (score_cols * 0.27) + (score_border * 0.18) + (score_center * 0.18) + (score_diag * 0.10), 6)

    score_mod3 = _distribution_score(_mod_counts(selected, 3), model.medians["mod3"], scale=0.9)
    score_mod4 = _distribution_score(_mod_counts(selected, 4), model.medians["mod4"], scale=0.9)
    score_mod5 = _distribution_score(_mod_counts(selected, 5), model.medians["mod5"], scale=0.9)
    finais = [0 for _ in range(10)]
    for n in selected:
        finais[n % 10] += 1
    score_finais = _score_closeness(max(finais), 2.0, 1.6)
    score_residuos_modulares = round((score_mod3 * 0.26) + (score_mod4 * 0.26) + (score_mod5 * 0.26) + (score_finais * 0.22), 6)

    score_regimes_historicos = round(_avg([model.number_regime_scores.get(n, 50.0) for n in selected]), 6)

    score_final = _safe_float(row, "score_final", 50.0)
    score_estatistico = _safe_float(row, "score_estatistico", score_final)
    score_transicao = _safe_float(row, "score_transicao", score_final)
    score_contextual = _safe_float(row, "score_contextual", _safe_float(row, "score_localidade_numerologia", 50.0))
    score_climatico = _safe_float(row, "score_climatico", 50.0)
    max_run = _safe_float(row, "maior_sequencia", 0.0)
    exact_seen = int(_safe_float(row, "ja_saiu_exatamente_no_historico", 0.0))
    false_positive_penalty = 0.0
    false_positive_penalty += 10.0 if exact_seen else 0.0
    false_positive_penalty += 11.0 if score_estatistico >= 98.0 and score_transicao < 72.0 else 0.0
    false_positive_penalty += 9.0 if score_contextual < 43.0 and score_climatico < 43.0 else 0.0
    false_positive_penalty += 8.0 if max_run >= 8.0 else 0.0
    false_positive_penalty += max(0.0, score_final - 92.0) * 0.35 if score_transicao < 70.0 else 0.0
    score_detector_falso_positivo = round(max(0.0, min(100.0, 100.0 - false_positive_penalty)), 6)

    score_hard_negative = round(
        (score_final * 0.44)
        + (score_combinatorio_avancado * 0.12)
        + (score_grafo_dezenas * 0.09)
        + (score_complemento_ausentes * 0.09)
        + (score_geometria_volante * 0.07)
        + (score_residuos_modulares * 0.06)
        + (score_regimes_historicos * 0.06)
        + (score_detector_falso_positivo * 0.07),
        6,
    )
    score_learning_to_rank = round(
        (score_hard_negative * 0.54)
        + (_safe_float(row, "score_decisao_protegida", score_final) * 0.18)
        + (score_transicao * 0.10)
        + (score_complemento_ausentes * 0.06)
        + (score_grafo_dezenas * 0.06)
        + (score_detector_falso_positivo * 0.06),
        6,
    )
    return {
        "score_combinatorio_avancado": score_combinatorio_avancado,
        "score_pares_atraso_freq": pair_score,
        "score_trios_atraso_freq": trio_score,
        "score_quartetos_atraso_freq": quartet_score,
        "score_grafo_dezenas": score_grafo_dezenas,
        "score_complemento_ausentes": score_complemento_ausentes,
        "score_geometria_volante": score_geometria_volante,
        "score_residuos_modulares": score_residuos_modulares,
        "score_regimes_historicos": score_regimes_historicos,
        "score_detector_falso_positivo": score_detector_falso_positivo,
        "score_hard_negative": score_hard_negative,
        "score_learning_to_rank": score_learning_to_rank,
        "linhas_top100": " ".join(str(v) for v in rows),
        "colunas_top100": " ".join(str(v) for v in cols),
        "borda_qtd": int(border_count),
        "centro_3x3_qtd": int(center_count),
        "diagonal_qtd": int(diag_count),
        "mod3": " ".join(str(v) for v in _mod_counts(selected, 3)),
        "mod4": " ".join(str(v) for v in _mod_counts(selected, 4)),
        "mod5": " ".join(str(v) for v in _mod_counts(selected, 5)),
    }


def _safe_float(row: Mapping[str, object], column: str, default: float = 0.0) -> float:
    try:
        value = row.get(column, default)  # type: ignore[attr-defined]
    except AttributeError:
        value = default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if pd.isna(out) else out


def _recent_low_frequency_numbers(concursos: pd.DataFrame, *, window: int = 30) -> List[int]:
    if concursos.empty:
        return list(NUMBERS)
    df = concursos.copy().sort_values("concurso").tail(int(window))
    counts: Counter[int] = Counter()
    for _, row in df.iterrows():
        counts.update(_nums_from_row(row))
    return [n for n, _count in sorted(((n, counts.get(n, 0)) for n in NUMBERS), key=lambda item: (item[1], item[0]))]


def _balanced_random_combo(rng: random.Random, *, forced: Sequence[int] = (), excluded: Sequence[int] = ()) -> Tuple[int, ...]:
    forced_set = {int(n) for n in forced}
    excluded_set = {int(n) for n in excluded} - forced_set
    available = [n for n in NUMBERS if n not in forced_set and n not in excluded_set]
    if len(forced_set) > 15 or len(available) < 15 - len(forced_set):
        available = [n for n in NUMBERS if n not in forced_set]
    for _attempt in range(250):
        nums = sorted(forced_set | set(rng.sample(available, 15 - len(forced_set))))
        even_count = sum(1 for n in nums if n % 2 == 0)
        total = sum(nums)
        rows, cols = _grid_counts(nums)
        if 5 <= even_count <= 9 and 165 <= total <= 225 and max(rows) <= 4 and max(cols) <= 4:
            return tuple(nums)
    return tuple(sorted(forced_set | set(rng.sample(available, 15 - len(forced_set)))))


def _build_coverage_hedge_candidates(
    concursos: pd.DataFrame,
    existing_nums: Sequence[str],
    *,
    top_count: int,
) -> pd.DataFrame:
    if int(top_count) < 50:
        return pd.DataFrame()
    existing = {str(value) for value in existing_nums}
    low_recent = _recent_low_frequency_numbers(concursos, window=30)
    last_concurso = int(concursos["concurso"].max()) if not concursos.empty else 0
    rng = random.Random(730000 + last_concurso)
    target_rows = max(1200, int(top_count) * 30)
    rows: List[Dict[str, object]] = []
    seen = set(existing)
    attempts = 0
    while len(rows) < target_rows and attempts < target_rows * 40:
        attempts += 1
        scenario = attempts % 6
        forced: List[int] = []
        excluded: List[int] = []
        score_bonus = 0.0
        label = "coverage_balanceado"
        if scenario == 0:
            excluded = [1]
            score_bonus = 2.6
            label = "coverage_sem_01"
        elif scenario == 1:
            excluded = [1, 22, 24, 25]
            forced = rng.sample(low_recent[:10], 3)
            score_bonus = 2.2
            label = "coverage_contrarian"
        elif scenario == 2:
            forced = rng.sample(low_recent[:12], 5)
            score_bonus = 2.0
            label = "coverage_baixa_freq_recente"
        elif scenario == 3:
            excluded = rng.sample([1, 21, 22, 24, 25], 2)
            forced = rng.sample(low_recent[:14], 4)
            score_bonus = 1.8
            label = "coverage_anti_saturacao"
        elif scenario == 4:
            excluded = [1]
            forced = rng.sample([n for n in low_recent[:16] if n != 1], 4)
            score_bonus = 2.4
            label = "coverage_sem_01_baixa_freq"
        combo = _balanced_random_combo(rng, forced=forced, excluded=excluded)
        nums_text = _format_nums(combo)
        if nums_text in seen:
            continue
        seen.add(nums_text)
        score_final = round(70.0 + score_bonus + rng.random() * 2.5, 6)
        rows.append(
            {
                "rank": 900000 + len(rows) + 1,
                "nums": nums_text,
                "score_final": score_final,
                "score_estatistico": score_final,
                "score_historico": 58.0 + score_bonus,
                "score_atraso": 62.0 + score_bonus,
                "score_combinatorio": 58.0,
                "score_localidade_numerologia": 50.0,
                "score_contextual": 50.0,
                "score_climatico": 50.0,
                "score_temporal_profundo": 54.0,
                "score_cenarios": 58.0,
                "score_contrarian": 74.0 + score_bonus,
                "score_transicao": 54.0,
                "score_decisao_protegida": score_final,
                "score_cobertura_risco_falso_negativo": 68.0 + score_bonus,
                "score_weights": "coverage_hedge",
                "source_model": label,
                "metodo": label,
                "concurso_base_final": last_concurso,
            }
        )
    return pd.DataFrame(rows)


def _candidate_number_strength(candidates: pd.DataFrame) -> Dict[int, float]:
    if candidates.empty or "nums" not in candidates.columns:
        return {n: 0.0 for n in NUMBERS}
    strength = {n: 0.0 for n in NUMBERS}
    counts = {n: 0 for n in NUMBERS}
    ranked = candidates.head(min(len(candidates), 3000)).copy().reset_index(drop=True)
    for idx, row in ranked.iterrows():
        try:
            nums = _parse_nums(str(row["nums"]))
        except ValueError:
            continue
        base = _safe_float(row, "score_final", _safe_float(row, "score_top100", 50.0))
        rank_weight = 1.0 / (1.0 + (float(idx) / 500.0))
        for n in nums:
            strength[int(n)] += base * rank_weight
            counts[int(n)] += 1
    return {n: round(strength[n] / max(1, counts[n]), 6) for n in NUMBERS}


def _ranked_universe_numbers(concursos: pd.DataFrame, candidates: pd.DataFrame) -> List[int]:
    strength = _candidate_number_strength(candidates)
    low_recent = _recent_low_frequency_numbers(concursos, window=30)
    low_recent_rank = {n: idx for idx, n in enumerate(low_recent)}
    last_draw = set(_nums_from_row(concursos.sort_values("concurso").iloc[-1])) if not concursos.empty else set()
    return sorted(
        NUMBERS,
        key=lambda n: (
            -strength.get(n, 0.0),
            0 if n in last_draw else 1,
            low_recent_rank.get(n, 999),
            n,
        ),
    )


def _closure_rows_for_universe(
    universe: Sequence[int],
    *,
    label: str,
    max_games: int,
    score_final: float,
    last_concurso: int,
    seen: set[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    try:
        games, coverage_pct, complete = greedy_reduced_closure(universe, guarantee_hits=14, max_games=max_games)
    except ValueError:
        return rows
    for game in games:
        nums_text = _format_nums(game)
        if nums_text in seen:
            continue
        seen.add(nums_text)
        rows.append(
            {
                "rank": 850000 + len(seen),
                "nums": nums_text,
                "score_final": float(score_final),
                "score_estatistico": float(score_final),
                "score_historico": 63.0,
                "score_atraso": 61.0,
                "score_combinatorio": 70.0,
                "score_localidade_numerologia": 50.0,
                "score_contextual": 52.0,
                "score_climatico": 50.0,
                "score_temporal_profundo": 58.0,
                "score_cenarios": 62.0,
                "score_contrarian": 60.0,
                "score_transicao": 64.0,
                "score_decisao_protegida": float(score_final),
                "score_cobertura_risco_falso_negativo": 72.0,
                "score_weights": f"{label}_coverage_pct={coverage_pct}_complete={int(complete)}",
                "source_model": label,
                "metodo": label,
                "concurso_base_final": last_concurso,
                "universo_fechamento_top100": _format_nums(universe),
                "cobertura_condicional_pct_top100": float(coverage_pct),
            }
        )
    return rows


def _build_closure_hedge_candidates(
    concursos: pd.DataFrame,
    candidates: pd.DataFrame,
    existing_nums: Sequence[str],
    *,
    top_count: int,
) -> pd.DataFrame:
    if int(top_count) < 50 or candidates.empty:
        return pd.DataFrame()
    last_concurso = int(concursos["concurso"].max()) if not concursos.empty else 0
    ranked_numbers = _ranked_universe_numbers(concursos, candidates)
    rank_order = {n: idx for idx, n in enumerate(ranked_numbers)}
    low_recent = _recent_low_frequency_numbers(concursos, window=30)
    last_draw = set(_nums_from_row(concursos.sort_values("concurso").iloc[-1])) if not concursos.empty else set()
    seen = {str(value) for value in existing_nums}
    universes: List[Tuple[str, List[int], int, float]] = []

    universes.append(("closure_universo_top20", sorted(ranked_numbers[:20]), 36, 77.0))
    universes.append(("closure_universo_top19", sorted(ranked_numbers[:19]), 28, 76.5))

    transition_universe = sorted(set(ranked_numbers[:12]) | last_draw | set(low_recent[:5]), key=lambda n: rank_order.get(n, 999))
    if len(transition_universe) >= 18:
        universes.append(("closure_transicao_20", sorted(transition_universe[:20]), 36, 76.8))

    contrarian_universe = sorted(set(ranked_numbers[:14]) | set(low_recent[:8]), key=lambda n: (0 if n in low_recent[:8] else 1, rank_order.get(n, 999)))
    if len(contrarian_universe) >= 18:
        universes.append(("closure_contrarian_20", sorted(contrarian_universe[:20]), 30, 75.8))

    no_one = [n for n in ranked_numbers if n != 1]
    if len(no_one) >= 19:
        universes.append(("closure_sem_01_19", sorted(no_one[:19]), 24, 75.5))

    rows: List[Dict[str, object]] = []
    for label, universe, max_games, score_final in universes:
        if len(universe) < 15:
            continue
        rows.extend(
            _closure_rows_for_universe(
                universe,
                label=label,
                max_games=max_games,
                score_final=score_final,
                last_concurso=last_concurso,
                seen=seen,
            )
        )
    return pd.DataFrame(rows)


def enrich_candidates_with_top100_scores(
    candidates: pd.DataFrame,
    concursos: pd.DataFrame,
    *,
    refinement_payload: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    prepared = candidates.copy().reset_index(drop=True)
    if "rank_top100_pool" in prepared.columns:
        prepared = prepared.drop(columns=["rank_top100_pool"])
    if "rank" in prepared.columns:
        prepared["rank"] = pd.to_numeric(prepared["rank"], errors="coerce")
        fallback_ranks = pd.Series(range(1, len(prepared) + 1), index=prepared.index, dtype="float64")
        prepared["rank"] = prepared["rank"].fillna(fallback_ranks).astype(int)
    else:
        prepared.insert(0, "rank", range(1, len(prepared) + 1))
    model = _build_advanced_model(concursos)
    guarded = enrich_candidates_with_decision_guard(prepared)
    rows: List[Dict[str, object]] = []
    for _, row in guarded.iterrows():
        nums = _parse_nums(str(row["nums"]))
        out = row.to_dict()
        out.update(_score_advanced_studies(nums, out, model))
        rows.append(out)
    enriched = pd.DataFrame(rows)
    enriched["score_top100"] = pd.to_numeric(enriched["score_learning_to_rank"], errors="coerce").fillna(0.0)
    enriched = apply_top50_refinement(enriched, refinement_payload, override_score_top100=True)
    enriched = enriched.sort_values(["score_top100", "score_final", "nums"], ascending=[False, False, True]).reset_index(drop=True)
    enriched.insert(0, "rank_top100_pool", range(1, len(enriched) + 1))
    return enriched


def _max_overlap_with_selected(nums: Sequence[int], selected_rows: Sequence[Mapping[str, object]]) -> int:
    nums_set = set(int(n) for n in nums)
    best = 0
    for row in selected_rows:
        other = set(_parse_nums(str(row["nums"])))
        best = max(best, len(nums_set & other))
    return int(best)


def _is_closure_candidate(row: Mapping[str, object]) -> bool:
    source = str(row.get("source_model", "") or "")
    weights = str(row.get("score_weights", "") or "")
    return source.startswith("closure_") or weights.startswith("closure_")


def _portfolio_adjusted_score(
    row: Mapping[str, object],
    nums: Sequence[int],
    *,
    selected_len: int,
    top_count: int,
    number_counts: Counter[int],
    first_counts: Counter[int],
) -> float:
    base = _safe_float(row, "score_top100", _safe_float(row, "score_final", 50.0))
    expected_next = float(selected_len + 1) * 15.0 / 25.0
    overuse_penalty = 0.0
    underuse_bonus = 0.0
    for n in nums:
        projected = float(number_counts[int(n)] + 1)
        overuse_penalty += max(0.0, projected - expected_next - 1.0) ** 1.48
        underuse_bonus += max(0.0, expected_next - projected)

    first = min(int(n) for n in nums)
    first_expected = _first_number_soft_target(selected_len + 1, first)
    first_penalty = max(0.0, float(first_counts[first] + 1) - first_expected - 1.0) ** 1.25
    first_bonus = max(0.0, first_expected - float(first_counts[first] + 1)) * 0.035

    final_cap = max(1.0, float(top_count) * 15.0 / 25.0)
    late_overuse = sum(max(0.0, float(number_counts[int(n)] + 1) - final_cap) for n in nums)
    closure_bonus = 0.0
    if _is_closure_candidate(row):
        coverage_pct = _safe_float(row, "cobertura_condicional_pct_top100", 0.0)
        closure_bonus = 5.5 + min(10.0, coverage_pct * 0.16)
    return round(
        base
        - (overuse_penalty * 1.08)
        - (first_penalty * 1.65)
        - (late_overuse * 0.55)
        + (underuse_bonus * 0.10)
        + first_bonus
        + closure_bonus,
        6,
    )


def _first_number_soft_target(position: int, first_number: int) -> float:
    ratios = {1: 0.28, 2: 0.23, 3: 0.18, 4: 0.14, 5: 0.12}
    return float(position) * ratios.get(int(first_number), 0.10)


def _first_number_cap(top_count: int, first_number: int, multiplier: float | None) -> int | None:
    if multiplier is None:
        return None
    soft_ratios = {1: 0.28, 2: 0.23, 3: 0.18, 4: 0.14, 5: 0.12}
    hard_ratios = {1: 0.34, 2: 0.28, 3: 0.22, 4: 0.17, 5: 0.15}
    first = int(first_number)
    soft_cap = ceil(float(top_count) * soft_ratios.get(first, 0.10) * float(multiplier))
    hard_cap = ceil(float(top_count) * hard_ratios.get(first, 0.12))
    return max(2, min(int(soft_cap), int(hard_cap)))


def select_top100_portfolio(candidates: pd.DataFrame, *, top_count: int, max_overlap: int = 13) -> pd.DataFrame:
    if candidates.empty:
        raise ValueError("Nenhum candidato disponivel para selecionar Top 100.")
    ranked = candidates.sort_values(["score_top100", "score_final", "nums"], ascending=[False, False, True]).reset_index(drop=True)
    selected: List[Dict[str, object]] = []
    selected_nums: set[str] = set()
    number_counts: Counter[int] = Counter()
    first_counts: Counter[int] = Counter()
    start_overlap = max(8, min(13, int(max_overlap)))
    stages = [
        (start_overlap, ceil(float(top_count) * 0.62), 1.00),
        (min(15, start_overlap + 1), ceil(float(top_count) * 0.66), 1.08),
        (min(15, start_overlap + 2), ceil(float(top_count) * 0.70), 1.16),
        (min(15, start_overlap + 3), ceil(float(top_count) * 0.73), 1.24),
        (15, ceil(float(top_count) * 0.76), 1.32),
    ]
    for overlap_limit, number_cap, first_cap_multiplier in stages:
        while len(selected) < int(top_count):
            best_row: Dict[str, object] | None = None
            best_score: float | None = None
            best_overlap = 0
            best_nums: Tuple[int, ...] | None = None
            for _, row in ranked.iterrows():
                nums_text = str(row["nums"])
                if nums_text in selected_nums:
                    continue
                nums = _parse_nums(nums_text)
                overlap = _max_overlap_with_selected(nums, selected)
                if selected and overlap > int(overlap_limit):
                    continue
                if number_cap is not None and any(number_counts[int(n)] >= int(number_cap) for n in nums):
                    continue
                first = min(int(n) for n in nums)
                first_cap = _first_number_cap(int(top_count), first, first_cap_multiplier)
                if first_cap is not None and first_counts[first] >= int(first_cap):
                    continue
                adjusted = _portfolio_adjusted_score(
                    row,
                    nums,
                    selected_len=len(selected),
                    top_count=int(top_count),
                    number_counts=number_counts,
                    first_counts=first_counts,
                )
                if best_score is None or adjusted > best_score:
                    best_score = adjusted
                    best_row = row.to_dict()
                    best_overlap = int(overlap)
                    best_nums = nums
            if best_row is None or best_nums is None:
                break
            best_row["max_overlap_top100_anterior"] = int(best_overlap)
            best_row["score_diversidade_top100"] = float(best_score or 0.0)
            best_row["qtd_dezenas_saturadas_antes"] = int(
                sum(1 for n in best_nums if number_cap is not None and number_counts[int(n)] >= max(0, int(number_cap) - 1))
            )
            best_first = min(int(n) for n in best_nums)
            best_row["primeira_dezena_top100"] = int(best_first)
            best_row["qtd_primeira_dezena_antes"] = int(first_counts[best_first])
            best_row["criterio_top100"] = (
                f"ranking_top100_diverso_overlap<={overlap_limit}"
                + (f"_cap_dezena<={int(number_cap)}" if number_cap is not None else "_sem_cap_dezena")
                + (
                    f"_cap_primeira_dezena<={_first_number_cap(int(top_count), best_first, first_cap_multiplier)}"
                    if first_cap_multiplier is not None
                    else "_sem_cap_primeira_dezena"
                )
            )
            selected.append(best_row)
            selected_nums.add(str(best_row["nums"]))
            number_counts.update(int(n) for n in best_nums)
            first_counts[best_first] += 1
        if len(selected) >= int(top_count):
            break
    while len(selected) < int(top_count):
        best_row = None
        best_score = None
        best_overlap = 0
        best_nums = None
        for _, row in ranked.iterrows():
            nums_text = str(row["nums"])
            if nums_text in selected_nums:
                continue
            nums = _parse_nums(nums_text)
            overlap = _max_overlap_with_selected(nums, selected)
            adjusted = _portfolio_adjusted_score(
                row,
                nums,
                selected_len=len(selected),
                top_count=int(top_count),
                number_counts=number_counts,
                first_counts=first_counts,
            )
            if best_score is None or adjusted > best_score:
                best_score = adjusted
                best_row = row.to_dict()
                best_overlap = int(overlap)
                best_nums = nums
        if best_row is None or best_nums is None:
            break
        best_first = min(int(n) for n in best_nums)
        best_row["max_overlap_top100_anterior"] = int(best_overlap)
        best_row["score_diversidade_top100"] = float(best_score or 0.0)
        best_row["qtd_dezenas_saturadas_antes"] = 0
        best_row["primeira_dezena_top100"] = int(best_first)
        best_row["qtd_primeira_dezena_antes"] = int(first_counts[best_first])
        best_row["criterio_top100"] = "ranking_top100_preenchimento_final_diverso"
        selected.append(best_row)
        selected_nums.add(str(best_row["nums"]))
        number_counts.update(int(n) for n in best_nums)
        first_counts[best_first] += 1
    out_df = pd.DataFrame(selected).head(int(top_count)).reset_index(drop=True)
    out_df.insert(0, "rank_top100", range(1, len(out_df) + 1))
    out_df["grupo_top"] = out_df["rank_top100"].map(lambda rank: "top10" if int(rank) <= 10 else ("top50" if int(rank) <= 50 else "top100"))
    if "source_model" in out_df.columns:
        out_df["estrategia_origem_top100"] = out_df["source_model"].fillna(TOP100_SOURCE_MODEL).astype(str)
    else:
        out_df["estrategia_origem_top100"] = TOP100_SOURCE_MODEL
    out_df["source_model"] = TOP100_SOURCE_MODEL
    out_df["metodo"] = TOP100_SOURCE_MODEL
    return out_df


def _build_report(
    final_games: pd.DataFrame,
    *,
    summary: Top100Summary,
    target_context: object,
    weights: Mapping[str, float],
    refinement_payload: Mapping[str, object] | None = None,
) -> str:
    refinement_metrics = refinement_payload.get("metrics", {}) if isinstance(refinement_payload, dict) and isinstance(refinement_payload.get("metrics"), dict) else {}
    lines = [
        "# Relatorio tecnico - Motor Top 100 / Top 50",
        "",
        f"Gerado em: {summary.generated_at}",
        f"Concurso-alvo: {summary.concurso_alvo}",
        f"Data do proximo concurso: {summary.data_proximo_concurso}",
        f"Metodo: {summary.metodo}",
        f"Top solicitado: {summary.top_count}",
        f"Pool analisado: {summary.top_pool}",
        f"Pesos supervisionados: {format_exhaustive_weights(resolve_exhaustive_weights(weights))}",
        f"Refinador Top50 aplicado: {'sim' if refinement_payload else 'nao'}",
        f"Refinador Top50 modelo: {refinement_payload.get('model', '-') if isinstance(refinement_payload, dict) else '-'}",
        f"Refinador Top50 rank medio historico antes/depois: {refinement_metrics.get('rank_before_avg', '-')} / {refinement_metrics.get('rank_after_avg', '-')}",
        f"Refinador Top50 Hit@50 historico antes/depois: {refinement_metrics.get('hit_top50_before', '-')}% / {refinement_metrics.get('hit_top50_after', '-')}%",
        "",
        "## Contexto",
        "",
        f"- Dia da semana: {getattr(target_context, 'dia_semana_nome', '-')}",
        f"- Lua: {getattr(target_context, 'fase_lua', '-')} ({float(getattr(target_context, 'iluminacao_lua_percentual', 0.0)):.2f}% iluminada)",
        f"- Numerologia data: {getattr(target_context, 'numerologia_data_raiz', '-')}",
        f"- Localidade: {getattr(target_context, 'local_sorteio_assumido', '-')}; {getattr(target_context, 'cidade_sorteio_assumida', '-')}/{getattr(target_context, 'uf_sorteio_assumida', '-')}",
        f"- Clima assinatura: {getattr(target_context, 'clima_assinatura', '-')}",
        "",
        "## Estudos adicionais desta camada",
        "",
        "1. objetivo Top 100 / Top 50 / Top 10;",
        "2. hard negatives: os jogos competem contra candidatos fortes do motor exaustivo, nao contra amostras faceis;",
        "3. pares, trios e quartetos por frequencia e atraso;",
        "4. grafo de dezenas por densidade e centralidade de pares;",
        "5. complemento das 10 ausentes;",
        "6. geometria do volante 5x5;",
        "7. residuos matematicos mod 3, mod 4, mod 5 e finais;",
        "8. regimes historicos recentes e longos;",
        "9. detector de falso positivo;",
        "10. score learning-to-rank interno para reordenar os candidatos;",
        "11. refinador Top50 pos-erro, quando treinado, para subir gabaritos historicos contra falsos positivos.",
        "",
        "## Top 100",
        "",
    ]
    for _, row in final_games.iterrows():
        lines.append(
            f"{int(row['rank_top100']):03d}. {row['nums']} | grupo={row['grupo_top']} | "
            f"score_top100={float(row['score_top100']):.6f} | score_base={float(row.get('score_final', 0.0)):.6f} | "
            f"score_refinado={float(row.get('score_top50_refinado', row.get('score_top100', 0.0))):.6f} | "
            f"comb_avancado={float(row.get('score_combinatorio_avancado', 0.0)):.2f} | "
            f"grafo={float(row.get('score_grafo_dezenas', 0.0)):.2f} | complemento={float(row.get('score_complemento_ausentes', 0.0)):.2f} | "
            f"geometria={float(row.get('score_geometria_volante', 0.0)):.2f} | falso_positivo={float(row.get('score_detector_falso_positivo', 0.0)):.2f}"
        )
    lines.extend(
        [
            "",
            "## Limite tecnico",
            "",
            f"A Lotofacil tem {TOTAL_COMBINATIONS} combinacoes possiveis de 15 dezenas. Este ranking aumenta disciplina e cobertura, mas nao garante que o resultado real esteja no Top 100.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_top100_prediction(
    concursos: pd.DataFrame,
    *,
    existing_candidates: pd.DataFrame | None,
    top_count: int,
    top_pool: int,
    max_overlap: int,
    draw_hour: int,
    draw_minute: int,
    exhaustive_limit: int | None,
    climate_features: pd.DataFrame | None,
    target_climate: Mapping[str, object] | None,
    weights: Mapping[str, float] | None,
    refinement_payload: Mapping[str, object] | None = None,
    prediction_csv_path: Path,
    report_path: Path,
    excel_path: Path,
) -> Top100Summary:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    last_concurso = int(df.iloc[-1]["concurso"])
    target_context = build_target_context(df, draw_hour=draw_hour, draw_minute=draw_minute, target_climate=target_climate)
    resolved_weights = resolve_exhaustive_weights(weights)
    analysis_pool = _expanded_top_pool(int(top_count), int(top_pool))
    expected_weights = format_exhaustive_weights(resolved_weights)
    candidates = existing_candidates.copy() if existing_candidates is not None and not existing_candidates.empty else pd.DataFrame()
    valid_cached = False
    if not candidates.empty and {"nums", "score_final", "score_weights", "concurso_base_final", "contexto_data_proximo_concurso"}.issubset(candidates.columns):
        base = pd.to_numeric(candidates["concurso_base_final"], errors="coerce").max()
        valid_cached = bool(
            pd.notna(base)
            and int(base) == last_concurso
            and str(candidates["contexto_data_proximo_concurso"].iloc[0]) == target_context.data_proximo_concurso
            and str(candidates["score_weights"].iloc[0]) == expected_weights
            and len(candidates) >= int(analysis_pool)
        )
    if not valid_cached:
        candidates, _summary = build_exhaustive_candidates(
            df,
            top_games=max(int(analysis_pool), int(top_count)),
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            limit_combinations=exhaustive_limit,
            weights=resolved_weights,
            climate_features=climate_features,
            target_climate=target_climate,
        )
    base_pool = candidates.head(int(analysis_pool)).copy()
    hedge_pool = _build_coverage_hedge_candidates(df, base_pool["nums"].astype(str).tolist() if "nums" in base_pool.columns else [], top_count=int(top_count))
    closure_pool = _build_closure_hedge_candidates(
        df,
        base_pool,
        list(base_pool["nums"].astype(str)) + (list(hedge_pool["nums"].astype(str)) if "nums" in hedge_pool.columns else []),
        top_count=int(top_count),
    )
    scored_pool = pd.concat([base_pool, hedge_pool, closure_pool], ignore_index=True).drop_duplicates(subset=["nums"], keep="first")
    enriched = enrich_candidates_with_top100_scores(scored_pool, df, refinement_payload=refinement_payload)
    final_games = select_top100_portfolio(enriched, top_count=int(top_count), max_overlap=int(max_overlap))
    generated_at = datetime.now().isoformat(timespec="seconds")
    final_games.insert(0, "generated_at", generated_at)
    final_games.insert(1, "concurso_alvo", last_concurso + 1)
    final_games.insert(2, "data_proximo_concurso", target_context.data_proximo_concurso)
    final_games["aviso"] = "Ranking estatistico; nao existe garantia de acerto em sorteios aleatorios."
    final_games = sanitize_dataframe_for_tabular_output(final_games)
    prediction_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    final_games.to_csv(prediction_csv_path, index=False, encoding="utf-8-sig")
    summary = Top100Summary(
        concurso_alvo=last_concurso + 1,
        generated_at=generated_at,
        top_count=int(top_count),
        top_pool=int(analysis_pool),
        selected_rows=int(len(final_games)),
        data_proximo_concurso=target_context.data_proximo_concurso,
        prediction_csv_path=str(prediction_csv_path),
        report_path=str(report_path),
        excel_path=str(excel_path),
    )
    report_path.write_text(
        _build_report(
            final_games,
            summary=summary,
            target_context=target_context,
            weights=resolved_weights,
            refinement_payload=refinement_payload,
        ),
        encoding="utf-8",
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        final_games.to_excel(writer, index=False, sheet_name="top100")
        final_games.head(50).to_excel(writer, index=False, sheet_name="top50")
        final_games.head(10).to_excel(writer, index=False, sheet_name="top10")
    return summary


def _actual_candidate_row(
    train_df: pd.DataFrame,
    target_row: pd.Series,
    *,
    climate_features: pd.DataFrame | None,
    draw_hour: int,
    draw_minute: int,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    actual = _nums_from_row(target_row)
    scored = _score_candidate_table(
        train_df=train_df,
        target_row=target_row,
        candidates=[actual],
        climate_features=climate_features,
        draw_hour=draw_hour,
        draw_minute=draw_minute,
    )
    scored = _apply_weights(scored, weights, column="score_final")
    scored["source_model"] = "gabarito_diagnostico_top100"
    scored["metodo"] = "gabarito_diagnostico_top100"
    return scored


def run_top100_backtest(
    concursos: pd.DataFrame,
    *,
    climate_features: pd.DataFrame | None,
    n_eval: int,
    min_history: int,
    top_count: int,
    top_pool: int,
    max_overlap: int,
    draw_hour: int,
    draw_minute: int,
    exhaustive_limit: int | None,
    weights: Mapping[str, float] | None,
    refinement_payload: Mapping[str, object] | None = None,
    results_csv_path: Path,
    summary_csv_path: Path,
    excel_path: Path,
) -> Top100BacktestSummary:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    if len(df) <= int(min_history):
        raise ValueError("Historico insuficiente para backtest Top 100.")
    resolved_weights = resolve_exhaustive_weights(weights)
    analysis_pool = _expanded_top_pool(int(top_count), int(top_pool))
    start_idx = max(int(min_history), len(df) - int(n_eval))
    rows: List[Dict[str, object]] = []
    for idx in range(start_idx, len(df)):
        train_df = df.iloc[:idx].copy()
        target_row = df.iloc[idx]
        actual_nums = _nums_from_row(target_row)
        actual_text = _format_nums(actual_nums)
        target_climate = None
        if climate_features is not None and not climate_features.empty and "concurso" in climate_features.columns:
            climate_df = climate_features.copy()
            climate_df["concurso"] = pd.to_numeric(climate_df["concurso"], errors="coerce")
            match = climate_df[climate_df["concurso"] == int(target_row["concurso"])]
            target_climate = match.iloc[0].to_dict() if not match.empty else None
        candidates, _summary = build_exhaustive_candidates(
            train_df,
            top_games=max(int(analysis_pool), int(top_count)),
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            limit_combinations=exhaustive_limit,
            weights=resolved_weights,
            climate_features=climate_features,
            target_climate=target_climate,
        )
        base_pool = candidates.head(int(analysis_pool)).copy()
        hedge_pool = _build_coverage_hedge_candidates(
            train_df,
            base_pool["nums"].astype(str).tolist() if "nums" in base_pool.columns else [],
            top_count=int(top_count),
        )
        closure_pool = _build_closure_hedge_candidates(
            train_df,
            base_pool,
            list(base_pool["nums"].astype(str)) + (list(hedge_pool["nums"].astype(str)) if "nums" in hedge_pool.columns else []),
            top_count=int(top_count),
        )
        scored_pool = pd.concat([base_pool, hedge_pool, closure_pool], ignore_index=True).drop_duplicates(subset=["nums"], keep="first")
        enriched = enrich_candidates_with_top100_scores(scored_pool, train_df, refinement_payload=refinement_payload)
        selected = select_top100_portfolio(enriched, top_count=int(top_count), max_overlap=int(max_overlap))
        selected_nums = set(str(value) for value in selected["nums"].tolist())
        actual_row = _actual_candidate_row(
            train_df,
            target_row,
            climate_features=climate_features,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            weights=resolved_weights,
        )
        diagnostic_pool = pd.concat([enriched, actual_row], ignore_index=True)
        diagnostic_ranked = enrich_candidates_with_top100_scores(
            diagnostic_pool,
            train_df,
            refinement_payload=refinement_payload,
        ).sort_values(
            ["score_top100", "score_final", "nums"], ascending=[False, False, True]
        ).reset_index(drop=True)
        matches = diagnostic_ranked.index[diagnostic_ranked["nums"] == actual_text].tolist()
        rank_diag = int(matches[0] + 1) if matches else 0
        rows.append(
            {
                "concurso": int(target_row["concurso"]),
                "data_sorteio": str(target_row["data_sorteio"]),
                "jogo_real": actual_text,
                "hit_top10": int(actual_text in set(str(v) for v in selected.head(10)["nums"].tolist())),
                "hit_top50": int(actual_text in set(str(v) for v in selected.head(50)["nums"].tolist())),
                "hit_top100": int(actual_text in selected_nums),
                "rank_diagnostico_com_gabarito": int(rank_diag),
                "top_count": int(top_count),
                "top_pool": int(analysis_pool),
                "exhaustive_limit": int(exhaustive_limit) if exhaustive_limit is not None else "",
            }
        )
    results = pd.DataFrame(rows)
    total = max(1, len(results))
    summary_rows = [
        {"metrica": "concursos_avaliados", "valor": int(len(results))},
        {"metrica": "hit_top10", "valor": int(results["hit_top10"].sum()) if not results.empty else 0},
        {"metrica": "hit_top50", "valor": int(results["hit_top50"].sum()) if not results.empty else 0},
        {"metrica": "hit_top100", "valor": int(results["hit_top100"].sum()) if not results.empty else 0},
        {"metrica": "taxa_top10", "valor": round(float(results["hit_top10"].sum()) / total * 100.0, 6) if not results.empty else 0.0},
        {"metrica": "taxa_top50", "valor": round(float(results["hit_top50"].sum()) / total * 100.0, 6) if not results.empty else 0.0},
        {"metrica": "taxa_top100", "valor": round(float(results["hit_top100"].sum()) / total * 100.0, 6) if not results.empty else 0.0},
        {"metrica": "rank_diagnostico_medio", "valor": round(float(pd.to_numeric(results["rank_diagnostico_com_gabarito"], errors="coerce").mean()), 6) if not results.empty else 0.0},
        {"metrica": "observacao", "valor": "hits_topN medem se a sequencia real apareceu no ranking gerado sem gabarito; rank_diagnostico injeta o gabarito apenas para medir capacidade de score."},
    ]
    summary = pd.DataFrame(summary_rows)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_csv_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results.to_excel(writer, index=False, sheet_name="resultados")
        summary.to_excel(writer, index=False, sheet_name="resumo")
    return Top100BacktestSummary(
        concursos_avaliados=int(len(results)),
        top_count=int(top_count),
        top_pool=int(analysis_pool),
        hit_top10=int(results["hit_top10"].sum()) if not results.empty else 0,
        hit_top50=int(results["hit_top50"].sum()) if not results.empty else 0,
        hit_top100=int(results["hit_top100"].sum()) if not results.empty else 0,
        taxa_top10=round(float(results["hit_top10"].sum()) / total * 100.0, 6) if not results.empty else 0.0,
        taxa_top50=round(float(results["hit_top50"].sum()) / total * 100.0, 6) if not results.empty else 0.0,
        taxa_top100=round(float(results["hit_top100"].sum()) / total * 100.0, 6) if not results.empty else 0.0,
        rank_diagnostico_medio=round(float(pd.to_numeric(results["rank_diagnostico_com_gabarito"], errors="coerce").mean()), 6) if not results.empty else 0.0,
        results_csv_path=str(results_csv_path),
        summary_csv_path=str(summary_csv_path),
        excel_path=str(excel_path),
    )
