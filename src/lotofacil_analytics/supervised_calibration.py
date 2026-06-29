from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from .exhaustive_optimizer import (
    DEFAULT_EXHAUSTIVE_WEIGHTS,
    FULL_EVEN_COUNT,
    FULL_SUM,
    NUMBERS,
    PAIR_COUNT,
    _climate_number_scores,
    _column_counts_from_omitted,
    _common_range_signatures,
    _context_number_scores,
    _contrarian_score,
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
from .storage import sanitize_dataframe_for_tabular_output
from .temporal_deep import temporal_deep_number_scores
from .transition_analysis import build_transition_model, score_transition_from_omitted


COMPONENTS = tuple(DEFAULT_EXHAUSTIVE_WEIGHTS.keys())


@dataclass(frozen=True)
class SupervisedCalibrationSummary:
    status: str
    contests_processed_this_run: int
    total_contests_processed: int
    current_concurso: int | None
    last_concurso: int | None
    rank_before_avg: float
    rank_after_avg: float
    weights_json_path: str
    state_json_path: str
    results_csv_path: str
    summary_csv_path: str
    weights_csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Calibracao Supervisionada",
                f"Status: {self.status}",
                f"Concursos processados nesta execucao: {self.contests_processed_this_run}",
                f"Concursos processados no total: {self.total_contests_processed}",
                f"Concurso atual: {self.current_concurso if self.current_concurso is not None else 'nenhum'}",
                f"Ultimo concurso processado: {self.last_concurso if self.last_concurso is not None else 'nenhum'}",
                f"Rank medio antes: {self.rank_before_avg:.4f}",
                f"Rank medio depois: {self.rank_after_avg:.4f}",
                f"Pesos aplicados no motor principal: {self.weights_json_path}",
                f"Estado retomavel: {self.state_json_path}",
                f"CSV resultados: {self.results_csv_path}",
                f"CSV resumo: {self.summary_csv_path}",
                f"CSV pesos: {self.weights_csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: usa o gabarito somente em concursos historicos e grava a media aprendida para concursos futuros.",
            ]
        )


def _now() -> str:
    return pd.Timestamp.now().isoformat(timespec="seconds")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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


def _nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _mask(nums: Sequence[int]) -> int:
    out = 0
    for n in nums:
        out |= 1 << (int(n) - 1)
    return int(out)


def _candidate_from_mask(mask: int) -> Tuple[int, ...]:
    return tuple(n for n in NUMBERS if int(mask) & (1 << (int(n) - 1)))


def _sample_candidates(actual: Sequence[int], *, samples: int, seed: int) -> List[Tuple[int, ...]]:
    rng = random.Random(int(seed))
    selected: Dict[int, Tuple[int, ...]] = {_mask(actual): tuple(sorted(int(n) for n in actual))}
    numbers = list(NUMBERS)
    target = max(1, int(samples))
    while len(selected) < target + 1:
        candidate = tuple(sorted(rng.sample(numbers, 15)))
        selected.setdefault(_mask(candidate), candidate)
    return list(selected.values())


def _percentile_rank(values: Sequence[float], actual_value: float) -> float:
    if not values:
        return 50.0
    lower_or_equal = sum(1 for value in values if float(value) <= float(actual_value))
    return round(float(lower_or_equal) / float(len(values)) * 100.0, 6)


def _rank_for_scores(rows: pd.DataFrame, *, score_column: str, actual_nums: Sequence[int]) -> Tuple[int, float, float]:
    if rows.empty:
        return 0, 0.0, 0.0
    actual_text = _format_nums(actual_nums)
    ranked = rows.copy()
    ranked[score_column] = pd.to_numeric(ranked[score_column], errors="coerce").fillna(0.0)
    ranked = ranked.sort_values([score_column, "nums"], ascending=[False, True]).reset_index(drop=True)
    matches = ranked.index[ranked["nums"] == actual_text].tolist()
    if not matches:
        return 0, 0.0, 0.0
    rank = int(matches[0] + 1)
    score = float(ranked.loc[matches[0], score_column])
    percentile = round((1.0 - ((rank - 1) / max(1, len(ranked) - 1))) * 100.0, 6)
    return rank, percentile, score


def _range_counts_from_selected(selected: Sequence[int]) -> List[int]:
    counts = [0, 0, 0, 0, 0]
    for n in selected:
        counts[(int(n) - 1) // 5] += 1
    return counts


def _climate_by_concurso(climate_features: pd.DataFrame | None) -> Dict[int, Mapping[str, object]]:
    if climate_features is None or climate_features.empty or "concurso" not in climate_features.columns:
        return {}
    df = climate_features.copy()
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")
    return {
        int(row["concurso"]): row.to_dict()
        for _, row in df.dropna(subset=["concurso"]).iterrows()
    }


def _score_candidate_components(
    selected: Sequence[int],
    *,
    train_df: pd.DataFrame,
    profile: Mapping[str, float],
    last_draw: set[int],
    recent_freq: Mapping[int, int],
    delays: Mapping[int, int],
    pair_median: float,
    pair_matrix: List[List[float]],
    incident_pair_sum: Mapping[int, float],
    total_pair_sum: float,
    common_signatures: set[str],
    context_scores: Mapping[int, float],
    climate_scores: Mapping[int, float],
    temporal_deep_scores: Mapping[int, float],
    transition_model: object,
    existing_draws: set[Tuple[int, ...]],
) -> Dict[str, float]:
    selected = tuple(sorted(int(n) for n in selected))
    omitted = tuple(n for n in NUMBERS if n not in set(selected))
    omitted_set = set(omitted)
    selected_sum = FULL_SUM - sum(omitted)
    selected_pairs = FULL_EVEN_COUNT - sum(1 for n in omitted if int(n) % 2 == 0)
    ranges = _range_counts_from_selected(selected)
    columns = _column_counts_from_omitted(omitted)
    max_run = _max_run_from_omitted(omitted_set)
    overlap_last = 15 - sum(1 for n in omitted if int(n) in last_draw)
    diagonal_strength = _diagonal_strength_from_omitted(omitted_set)
    signature = "-".join(str(value) for value in ranges)

    stat_penalty = 0.0
    stat_penalty += _distance_outside_band(selected_sum, profile["sum_p10"], profile["sum_p90"]) / 1.6
    stat_penalty += _distance_outside_band(selected_pairs, profile["pairs_p10"], profile["pairs_p90"]) * 3.0
    stat_penalty += 0.0 if signature in common_signatures else sum(abs(value - 3) for value in ranges) * 1.8
    stat_penalty += abs(overlap_last - profile["median_overlap"]) * 2.2
    stat_penalty += max(0.0, max_run - profile["run_p95"]) * 2.2
    score_estatistico = _score_0_100(stat_penalty)

    recent_total = sum(float(recent_freq.get(n, 0)) for n in NUMBERS)
    selected_recent_sum = sum(float(recent_freq.get(int(n), 0)) for n in selected)
    avg_recent = selected_recent_sum / 15.0
    expected_recent = recent_total / max(1, len(NUMBERS))
    score_historico = _score_0_100(abs(avg_recent - expected_recent) * 1.2)

    delay_values = list(delays.values())
    median_delay = float(pd.Series(delay_values, dtype="float64").median()) if delay_values else 0.0
    selected_delay_sum = sum(float(delays.get(int(n), 0)) for n in selected)
    avg_delay = selected_delay_sum / 15.0
    score_atraso = _score_0_100(abs(avg_delay - median_delay) * 1.4)

    selected_pair_sum = _pair_selected_sum_from_omitted(
        total_pair_sum=total_pair_sum,
        incident_pair_sum=dict(incident_pair_sum),
        pair_matrix=pair_matrix,
        omitted=omitted,
    )
    avg_pair_freq = selected_pair_sum / PAIR_COUNT
    score_combinatorio = _score_0_100(max(0.0, avg_pair_freq - pair_median) / 12.0)

    score_contextual = round(max(0.0, min(100.0, sum(float(context_scores.get(n, 50.0)) for n in selected) / 15.0)), 6)
    score_climatico = round(max(0.0, min(100.0, sum(float(climate_scores.get(n, 50.0)) for n in selected) / 15.0)), 6)
    score_temporal = round(max(0.0, min(100.0, sum(float(temporal_deep_scores.get(n, 50.0)) for n in selected) / 15.0)), 6)
    score_cenarios = _scenario_score(
        total=selected_sum,
        max_run=max_run,
        ranges=ranges,
        columns=columns,
        diagonal_strength=diagonal_strength,
        profile=dict(profile),
        common_signatures=common_signatures,
        selected=selected,
    )
    score_contrarian = _contrarian_score(selected, max_run=max_run, ranges=ranges, columns=columns)
    transition = score_transition_from_omitted(omitted, transition_model)
    score_transicao = float(transition["score_transicao"])
    score_nao_repeticao = 100.0 if tuple(selected) not in existing_draws else 92.0
    return {
        "estatistico": float(score_estatistico),
        "historico": float(score_historico),
        "atraso": float(score_atraso),
        "combinatorio": float(score_combinatorio),
        "localidade_numerologia": float(score_contextual),
        "climatico": float(score_climatico),
        "temporal_profundo": float(score_temporal),
        "cenarios": float(score_cenarios),
        "contrarian": float(score_contrarian),
        "transicao": float(score_transicao),
        "nao_repeticao_exata": float(score_nao_repeticao),
    }


def _score_candidate_table(
    *,
    train_df: pd.DataFrame,
    target_row: pd.Series,
    candidates: Sequence[Sequence[int]],
    climate_features: pd.DataFrame | None,
    draw_hour: int,
    draw_minute: int,
) -> pd.DataFrame:
    draws = [_nums_from_row(row) for _, row in train_df.iterrows()]
    profile = _historical_profile(draws)
    last_draw = set(draws[-1])
    recent_freq = _recent_freq(draws, window=100)
    delays = _delays(draws)
    pair_freq = _pair_counter(draws)
    common_signatures = _common_range_signatures(draws)
    climate_map = _climate_by_concurso(climate_features)
    context_model = build_context_model(
        train_df,
        draw_hour=draw_hour,
        draw_minute=draw_minute,
        climate_features=climate_features,
        target_climate=climate_map.get(int(target_row["concurso"])),
    )
    context_scores = _context_number_scores(context_model)
    climate_scores = _climate_number_scores(context_model)
    temporal_deep_scores = temporal_deep_number_scores(train_df, target_date=str(target_row["data_sorteio"]))
    transition_model = build_transition_model(train_df)

    pair_values = list(pair_freq.values())
    pair_median = float(pd.Series(pair_values, dtype="float64").median()) if pair_values else 0.0
    pair_matrix = [[0.0 for _ in range(max(NUMBERS) + 1)] for _ in range(max(NUMBERS) + 1)]
    for (a, b), value in pair_freq.items():
        pair_matrix[int(a)][int(b)] = float(value)
        pair_matrix[int(b)][int(a)] = float(value)
    total_pair_sum = sum(pair_matrix[a][b] for a, b in combinations(NUMBERS, 2))
    incident_pair_sum = {
        n: sum(pair_matrix[n][other] for other in NUMBERS if other != n)
        for n in NUMBERS
    }
    existing_draws = {tuple(draw) for draw in draws}

    rows: List[Dict[str, object]] = []
    for candidate in candidates:
        scores = _score_candidate_components(
            candidate,
            train_df=train_df,
            profile=profile,
            last_draw=last_draw,
            recent_freq=recent_freq,
            delays=delays,
            pair_median=pair_median,
            pair_matrix=pair_matrix,
            incident_pair_sum=incident_pair_sum,
            total_pair_sum=total_pair_sum,
            common_signatures=common_signatures,
            context_scores=context_scores,
            climate_scores=climate_scores,
            temporal_deep_scores=temporal_deep_scores,
            transition_model=transition_model,
            existing_draws=existing_draws,
        )
        row: Dict[str, object] = {"nums": _format_nums(candidate)}
        for component, value in scores.items():
            row[f"score_{component}"] = round(float(value), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_weights(rows: pd.DataFrame, weights: Mapping[str, float], *, column: str) -> pd.DataFrame:
    out = rows.copy()
    resolved = resolve_exhaustive_weights(weights)
    total = 0.0
    for component in COMPONENTS:
        total = total + float(resolved[component]) * pd.to_numeric(out[f"score_{component}"], errors="coerce").fillna(0.0)
    out[column] = total.round(6)
    return out


def _learn_weights_from_gabarito(rows: pd.DataFrame, actual_nums: Sequence[int]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    actual_text = _format_nums(actual_nums)
    actual_rows = rows[rows["nums"] == actual_text]
    if actual_rows.empty:
        return dict(DEFAULT_EXHAUSTIVE_WEIGHTS), {}, {}
    actual = actual_rows.iloc[0]
    edges: Dict[str, float] = {}
    percentiles: Dict[str, float] = {}
    raw: Dict[str, float] = {}
    for component in COMPONENTS:
        column = f"score_{component}"
        values = pd.to_numeric(rows[column], errors="coerce").fillna(0.0).tolist()
        actual_value = float(actual[column])
        median_value = float(pd.Series(values, dtype="float64").median()) if values else 50.0
        percentile = _percentile_rank(values, actual_value)
        edge = max(0.0, actual_value - median_value)
        edges[component] = round(float(edge), 6)
        percentiles[component] = round(float(percentile), 6)
        raw[component] = max(0.0, edge * max(0.0, percentile - 45.0))
    if sum(raw.values()) <= 0.0:
        fallback = {
            component: max(0.001, max(0.0, percentiles.get(component, 50.0) - 35.0))
            for component in COMPONENTS
        }
        raw = fallback
    return resolve_exhaustive_weights(raw), edges, percentiles


def _average_weights(results: pd.DataFrame) -> Dict[str, float]:
    if results.empty:
        return dict(DEFAULT_EXHAUSTIVE_WEIGHTS)
    totals = {component: 0.0 for component in COMPONENTS}
    total_weight = 0.0
    for _, row in results.iterrows():
        before = float(row.get("rank_antes", 0) or 0)
        after = float(row.get("rank_depois", 0) or 0)
        improvement = max(0.0, before - after)
        weight = 1.0 + improvement
        total_weight += weight
        for component in COMPONENTS:
            totals[component] += float(row.get(f"peso_{component}", DEFAULT_EXHAUSTIVE_WEIGHTS[component]) or 0.0) * weight
    if total_weight <= 0:
        return dict(DEFAULT_EXHAUSTIVE_WEIGHTS)
    return resolve_exhaustive_weights({component: totals[component] / total_weight for component in COMPONENTS})


def _write_outputs(
    *,
    results: pd.DataFrame,
    state: Mapping[str, object],
    summary_csv_path: Path,
    weights_csv_path: Path,
    weights_json_path: Path,
    excel_path: Path,
    from_concurso: int,
    to_concurso: int | None,
    samples: int,
    min_history: int,
    eligible_target_count: int,
    skipped_min_history_count: int,
    draw_hour: int,
    draw_minute: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    weights = _average_weights(results)
    summary_rows: List[Dict[str, object]] = []
    if not results.empty:
        rank_before = pd.to_numeric(results["rank_antes"], errors="coerce")
        rank_after = pd.to_numeric(results["rank_depois"], errors="coerce")
        summary_rows.extend(
            [
                {"metrica": "status", "valor": state.get("status", "")},
                {"metrica": "concursos_processados", "valor": int(results["concurso"].nunique())},
                {"metrica": "rank_antes_medio", "valor": round(float(rank_before.mean()), 6)},
                {"metrica": "rank_depois_medio", "valor": round(float(rank_after.mean()), 6)},
                {"metrica": "melhora_rank_media", "valor": round(float((rank_before - rank_after).mean()), 6)},
                {"metrica": "percentil_antes_medio", "valor": round(float(pd.to_numeric(results["percentil_antes"], errors="coerce").mean()), 6)},
                {"metrica": "percentil_depois_medio", "valor": round(float(pd.to_numeric(results["percentil_depois"], errors="coerce").mean()), 6)},
                {"metrica": "ultimo_concurso", "valor": int(pd.to_numeric(results["concurso"], errors="coerce").max())},
                {"metrica": "samples_por_concurso", "valor": int(samples)},
                {"metrica": "historico_minimo_supervisionado", "valor": int(min_history)},
                {"metrica": "concursos_elegiveis", "valor": int(eligible_target_count)},
                {"metrica": "concursos_pulados_por_historico_minimo", "valor": int(skipped_min_history_count)},
            ]
        )
    else:
        summary_rows.append({"metrica": "status", "valor": state.get("status", "sem_resultados")})
    summary = pd.DataFrame(summary_rows)
    weights_rows = [
        {
            "componente": component,
            "peso": round(float(weights[component]), 10),
            "peso_percentual": round(float(weights[component]) * 100.0, 6),
        }
        for component in COMPONENTS
    ]
    weights_df = pd.DataFrame(weights_rows)
    payload = {
        "model": "supervised_answer_key_calibration_v1",
        "source": "historical_gabarito_supervisionado",
        "updated_at": _now(),
        "from_concurso": int(from_concurso),
        "to_concurso": int(to_concurso) if to_concurso is not None else None,
        "samples_per_concurso": int(samples),
        "min_history": int(min_history),
        "eligible_contests": int(eligible_target_count),
        "skipped_min_history_contests": int(skipped_min_history_count),
        "draw_hour": int(draw_hour),
        "draw_minute": int(draw_minute),
        "contests": int(results["concurso"].nunique()) if not results.empty else 0,
        "weights": {component: round(float(weights[component]), 10) for component in COMPONENTS},
        "score_weights": format_exhaustive_weights(weights),
        "note": "Pesos medios aprendidos usando gabarito de concursos historicos; nao usa resultado futuro na geracao de novos jogos.",
    }
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    weights_csv_path.parent.mkdir(parents=True, exist_ok=True)
    weights_json_path.parent.mkdir(parents=True, exist_ok=True)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    weights_df.to_csv(weights_csv_path, index=False, encoding="utf-8-sig")
    weights_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="resumo")
        weights_df.to_excel(writer, index=False, sheet_name="pesos_medios")
        results.tail(1000).to_excel(writer, index=False, sheet_name="resultados")
    return summary, weights


def run_supervised_calibration(
    concursos: pd.DataFrame,
    *,
    climate_features: pd.DataFrame | None,
    from_concurso: int,
    to_concurso: int | None,
    samples: int,
    max_contests: int,
    seed: int,
    draw_hour: int,
    draw_minute: int,
    min_history: int,
    reset: bool,
    state_json_path: Path,
    results_csv_path: Path,
    summary_csv_path: Path,
    weights_csv_path: Path,
    excel_path: Path,
    weights_json_path: Path,
) -> SupervisedCalibrationSummary:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado.")
    if int(samples) <= 0:
        raise ValueError("--supervised-samples deve ser maior que zero.")

    if reset:
        for path in [state_json_path, results_csv_path, summary_csv_path, weights_csv_path, excel_path]:
            if path.exists():
                path.unlink()

    started = time.perf_counter()
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce").astype("Int64")
    max_target = int(to_concurso) if to_concurso is not None else int(df["concurso"].max())
    contest_indices: Dict[int, int] = {}
    for idx, value in df["concurso"].items():
        if pd.notna(value):
            contest_indices[int(value)] = int(idx)
    requested_targets = [
        int(value)
        for value in df["concurso"].dropna().astype(int).tolist()
        if int(value) >= int(from_concurso) and int(value) <= int(max_target)
    ]
    targets = [target for target in requested_targets if int(contest_indices.get(target, -1)) >= int(min_history)]
    target_set = set(targets)
    skipped_min_history_count = len(requested_targets) - len(targets)
    if not targets:
        raise ValueError("Nenhum concurso elegivel no intervalo da calibracao supervisionada com o historico minimo informado.")

    results = _read_csv(results_csv_path)
    processed = set(int(value) for value in pd.to_numeric(results.get("concurso", pd.Series(dtype=int)), errors="coerce").dropna().tolist()) if not results.empty else set()
    processed_in_scope = processed & target_set
    pending_in_scope = sorted(target_set - processed)
    state = _load_json(state_json_path)
    state.update(
        {
            "status": "running",
            "started_last_run_at": _now(),
            "from_concurso": int(from_concurso),
            "to_concurso": int(max_target),
            "samples": int(samples),
            "max_contests_this_run": int(max_contests),
            "min_history": int(min_history),
            "requested_target_count": int(len(requested_targets)),
            "eligible_target_count": int(len(targets)),
            "skipped_min_history_count": int(skipped_min_history_count),
            "first_eligible_concurso": int(min(targets)),
            "last_eligible_concurso": int(max(targets)),
            "processed_eligible_count": int(len(processed_in_scope)),
            "remaining_eligible_count": int(len(pending_in_scope)),
            "progress_percent": round(float(len(processed_in_scope)) / float(len(targets)) * 100.0, 6),
            "next_pending_concurso": int(pending_in_scope[0]) if pending_in_scope else None,
            "draw_hour": int(draw_hour),
            "draw_minute": int(draw_minute),
        }
    )
    _write_json(state_json_path, state)

    processed_this_run = 0
    for target_concurso in targets:
        if target_concurso in processed:
            continue
        matches = df.index[df["concurso"].astype(int) == int(target_concurso)].tolist()
        if not matches:
            continue
        target_idx = int(matches[0])
        if target_idx < int(min_history):
            continue
        target_row = df.iloc[target_idx]
        train_df = df.iloc[:target_idx].copy()
        actual_nums = _nums_from_row(target_row)
        candidates = _sample_candidates(actual_nums, samples=int(samples), seed=int(seed) + int(target_concurso) * 7919)
        scored = _score_candidate_table(
            train_df=train_df,
            target_row=target_row,
            candidates=candidates,
            climate_features=climate_features,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        current_weights = _average_weights(results)
        before_table = _apply_weights(scored, current_weights, column="score_antes")
        learned_weights, edges, percentiles = _learn_weights_from_gabarito(scored, actual_nums)
        after_table = _apply_weights(before_table, learned_weights, column="score_depois")
        rank_before, percentile_before, score_before = _rank_for_scores(after_table, score_column="score_antes", actual_nums=actual_nums)
        rank_after, percentile_after, score_after = _rank_for_scores(after_table, score_column="score_depois", actual_nums=actual_nums)
        actual_row = after_table[after_table["nums"] == _format_nums(actual_nums)].iloc[0]
        result_row: Dict[str, object] = {
            "concurso": int(target_concurso),
            "data_sorteio": str(target_row.get("data_sorteio", "")),
            "processed_at": _now(),
            "jogo_real": _format_nums(actual_nums),
            "samples": int(samples),
            "rank_antes": int(rank_before),
            "rank_depois": int(rank_after),
            "melhora_rank": int(rank_before - rank_after),
            "percentil_antes": float(percentile_before),
            "percentil_depois": float(percentile_after),
            "score_antes": round(float(score_before), 6),
            "score_depois": round(float(score_after), 6),
            "score_weights": format_exhaustive_weights(learned_weights),
        }
        for component in COMPONENTS:
            result_row[f"score_real_{component}"] = round(float(actual_row[f"score_{component}"]), 6)
            result_row[f"edge_{component}"] = round(float(edges.get(component, 0.0)), 6)
            result_row[f"percentil_{component}"] = round(float(percentiles.get(component, 0.0)), 6)
            result_row[f"peso_{component}"] = round(float(learned_weights[component]), 10)
        results = pd.concat([results, pd.DataFrame([result_row])], ignore_index=True)
        results = sanitize_dataframe_for_tabular_output(results)
        results_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_csv_path, index=False, encoding="utf-8-sig")
        processed.add(int(target_concurso))
        processed_this_run += 1
        summary, weights = _write_outputs(
            results=results,
            state=state,
            summary_csv_path=summary_csv_path,
            weights_csv_path=weights_csv_path,
            weights_json_path=weights_json_path,
            excel_path=excel_path,
            from_concurso=from_concurso,
            to_concurso=max_target,
            samples=samples,
            min_history=min_history,
            eligible_target_count=len(targets),
            skipped_min_history_count=skipped_min_history_count,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        processed_in_scope = processed & target_set
        pending_in_scope = sorted(target_set - processed)
        state.update(
            {
                "status": "running",
                "current_concurso": int(target_concurso),
                "last_concurso": int(target_concurso),
                "last_rank_antes": int(rank_before),
                "last_rank_depois": int(rank_after),
                "last_melhora_rank": int(rank_before - rank_after),
                "total_contests_processed": int(results["concurso"].nunique()),
                "contests_processed_this_run": int(processed_this_run),
                "processed_eligible_count": int(len(processed_in_scope)),
                "remaining_eligible_count": int(len(pending_in_scope)),
                "progress_percent": round(float(len(processed_in_scope)) / float(len(targets)) * 100.0, 6),
                "next_pending_concurso": int(pending_in_scope[0]) if pending_in_scope else None,
                "rank_before_avg": round(float(pd.to_numeric(results["rank_antes"], errors="coerce").mean()), 6),
                "rank_after_avg": round(float(pd.to_numeric(results["rank_depois"], errors="coerce").mean()), 6),
                "weights": {component: round(float(weights[component]), 10) for component in COMPONENTS},
                "updated_at": _now(),
                "elapsed_seconds_current_run": round(float(time.perf_counter() - started), 6),
            }
        )
        _write_json(state_json_path, state)
        if int(max_contests) > 0 and processed_this_run >= int(max_contests):
            state["status"] = "paused_by_contest_limit"
            state["updated_at"] = _now()
            _write_json(state_json_path, state)
            break

    remaining = [target for target in targets if target not in processed]
    if not remaining:
        state["status"] = "complete"
    elif processed_this_run == 0 and len(processed) >= len(targets):
        state["status"] = "complete"
    elif state.get("status") != "paused_by_contest_limit":
        state["status"] = "complete" if not remaining else "running"
    processed_in_scope = processed & target_set
    pending_in_scope = sorted(target_set - processed)
    state["processed_eligible_count"] = int(len(processed_in_scope))
    state["remaining_eligible_count"] = int(len(pending_in_scope))
    state["progress_percent"] = round(float(len(processed_in_scope)) / float(len(targets)) * 100.0, 6)
    state["next_pending_concurso"] = int(pending_in_scope[0]) if pending_in_scope else None
    state["updated_at"] = _now()
    state["elapsed_seconds_current_run"] = round(float(time.perf_counter() - started), 6)
    _write_json(state_json_path, state)
    summary, weights = _write_outputs(
        results=results,
        state=state,
        summary_csv_path=summary_csv_path,
        weights_csv_path=weights_csv_path,
        weights_json_path=weights_json_path,
        excel_path=excel_path,
        from_concurso=from_concurso,
        to_concurso=max_target,
        samples=samples,
        min_history=min_history,
        eligible_target_count=len(targets),
        skipped_min_history_count=skipped_min_history_count,
        draw_hour=draw_hour,
        draw_minute=draw_minute,
    )
    rank_before_avg = float(pd.to_numeric(results.get("rank_antes", pd.Series(dtype=float)), errors="coerce").mean()) if not results.empty else 0.0
    rank_after_avg = float(pd.to_numeric(results.get("rank_depois", pd.Series(dtype=float)), errors="coerce").mean()) if not results.empty else 0.0
    return SupervisedCalibrationSummary(
        status=str(state.get("status", "")),
        contests_processed_this_run=int(processed_this_run),
        total_contests_processed=int(results["concurso"].nunique()) if not results.empty else 0,
        current_concurso=int(state["current_concurso"]) if state.get("current_concurso") else None,
        last_concurso=int(state["last_concurso"]) if state.get("last_concurso") else None,
        rank_before_avg=round(float(rank_before_avg), 6) if pd.notna(rank_before_avg) else 0.0,
        rank_after_avg=round(float(rank_after_avg), 6) if pd.notna(rank_after_avg) else 0.0,
        weights_json_path=str(weights_json_path),
        state_json_path=str(state_json_path),
        results_csv_path=str(results_csv_path),
        summary_csv_path=str(summary_csv_path),
        weights_csv_path=str(weights_csv_path),
        excel_path=str(excel_path),
    )


def load_supervised_calibration_status(
    *,
    state_json_path: Path,
    results_csv_path: Path,
    summary_csv_path: Path,
    weights_csv_path: Path,
    weights_json_path: Path,
    recent_rows: int = 12,
) -> Dict[str, object]:
    state = _load_json(state_json_path)
    results = _read_csv(results_csv_path)
    summary = _read_csv(summary_csv_path)
    weights = _read_csv(weights_csv_path)
    weights_payload = _load_json(weights_json_path)
    recent = results.tail(int(recent_rows)) if not results.empty else pd.DataFrame()
    best = pd.DataFrame()
    blocks = pd.DataFrame()
    if not results.empty:
        scored = results.copy()
        scored["concurso"] = pd.to_numeric(scored["concurso"], errors="coerce")
        scored["rank_antes"] = pd.to_numeric(scored["rank_antes"], errors="coerce")
        scored["rank_depois"] = pd.to_numeric(scored["rank_depois"], errors="coerce")
        scored["melhora_rank"] = pd.to_numeric(scored["melhora_rank"], errors="coerce")
        scored["percentil_depois"] = pd.to_numeric(scored["percentil_depois"], errors="coerce")
        scored = scored.dropna(subset=["concurso"])
        state.setdefault("total_contests_processed", int(scored["concurso"].nunique()))
        state.setdefault("first_processed_concurso", int(scored["concurso"].min()))
        state.setdefault("last_processed_concurso", int(scored["concurso"].max()))
        if "rank_antes" in scored and "rank_depois" in scored:
            state.setdefault("rank_before_avg", round(float(scored["rank_antes"].mean()), 6))
            state.setdefault("rank_after_avg", round(float(scored["rank_depois"].mean()), 6))
            state["rank_improvement_avg"] = round(float((scored["rank_antes"] - scored["rank_depois"]).mean()), 6)
            state["best_rank_after"] = int(scored["rank_depois"].min()) if pd.notna(scored["rank_depois"].min()) else None
        best = scored.sort_values(["rank_depois", "rank_antes", "concurso"], ascending=[True, True, False]).head(10)
        scored["bloco_inicio"] = (((scored["concurso"].astype(int) - 1) // 100) * 100 + 1).astype(int)
        scored["bloco_fim"] = scored["bloco_inicio"] + 99
        block_rows: List[Dict[str, object]] = []
        for (start, end), group in scored.groupby(["bloco_inicio", "bloco_fim"], sort=True):
            block_rows.append(
                {
                    "bloco": f"{int(start)}-{int(end)}",
                    "concursos": int(group["concurso"].nunique()),
                    "rank_antes_medio": round(float(group["rank_antes"].mean()), 6),
                    "rank_depois_medio": round(float(group["rank_depois"].mean()), 6),
                    "melhora_media": round(float(group["melhora_rank"].mean()), 6),
                    "percentil_depois_medio": round(float(group["percentil_depois"].mean()), 6),
                }
            )
        blocks = pd.DataFrame(block_rows).tail(18)

    def records(frame: pd.DataFrame) -> List[Dict[str, object]]:
        if frame.empty:
            return []
        clean = frame.astype(object).where(pd.notna(frame), None)
        return clean.to_dict(orient="records")

    return {
        "state": state,
        "recent_results": records(recent),
        "best_results": records(best),
        "progress_blocks": records(blocks),
        "summary": records(summary),
        "weights": records(weights),
        "engine_weights": weights_payload.get("weights", {}) if isinstance(weights_payload, dict) else {},
        "paths": {
            "state_json_path": str(state_json_path),
            "results_csv_path": str(results_csv_path),
            "summary_csv_path": str(summary_csv_path),
            "weights_csv_path": str(weights_csv_path),
            "weights_json_path": str(weights_json_path),
        },
    }
