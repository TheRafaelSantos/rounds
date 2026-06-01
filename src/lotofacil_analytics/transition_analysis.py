from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .backtest_lotofacil import PICK_SIZE
from .normalize import DEZENAS
from .optimizer import MAX_DEZENA


NUMBERS = tuple(range(1, MAX_DEZENA + 1))
RANGE_BANDS: Tuple[Tuple[int, int], ...] = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25))
FULL_SUM = sum(NUMBERS)


@dataclass(frozen=True)
class TransitionModel:
    last_draw: Tuple[int, ...]
    last_draw_set: frozenset[int]
    last_sum: int
    median_repeats: float
    repeats_p10: float
    repeats_p90: float
    delta_sum_p10: float
    delta_sum_p90: float
    stay_probability: Dict[int, float]
    enter_probability: Dict[int, float]
    candidate_number_score: Dict[int, float]
    full_candidate_number_score: float
    last_range_counts: Tuple[int, ...]
    not_last_range_counts: Tuple[int, ...]
    common_repeat_signatures: frozenset[str]
    common_enter_signatures: frozenset[str]
    common_exit_signatures: frozenset[str]
    summary: Dict[str, object]


def _nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _range_index(n: int) -> int:
    return min(4, max(0, (int(n) - 1) // 5))


def _range_counts(nums: Sequence[int]) -> Tuple[int, ...]:
    counts = [0, 0, 0, 0, 0]
    for n in nums:
        counts[_range_index(int(n))] += 1
    return tuple(counts)


def _line_counts(nums: Sequence[int]) -> Tuple[int, ...]:
    return _range_counts(nums)


def _column_counts(nums: Sequence[int]) -> Tuple[int, ...]:
    counts = [0, 0, 0, 0, 0]
    for n in nums:
        counts[(int(n) - 1) % 5] += 1
    return tuple(counts)


def _signature(values: Sequence[int]) -> str:
    return "-".join(str(int(value)) for value in values)


def _percentile(values: Sequence[float], pct: float, default: float) -> float:
    if not values:
        return float(default)
    return float(pd.Series(values, dtype="float64").quantile(float(pct)))


def _median(values: Sequence[float], default: float) -> float:
    if not values:
        return float(default)
    return float(pd.Series(values, dtype="float64").median())


def _score_0_100(penalty: float) -> float:
    return round(max(0.0, min(100.0, 100.0 - float(penalty))), 6)


def _distance_outside_band(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _top_signatures(counter: Counter[str], *, min_fraction: float = 0.015, max_items: int = 30) -> frozenset[str]:
    total = sum(counter.values())
    if total <= 0:
        return frozenset()
    rows = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    selected = [key for key, value in rows if value / total >= min_fraction]
    if len(selected) < min(5, len(rows)):
        selected = [key for key, _value in rows[: min(5, len(rows))]]
    return frozenset(selected[:max_items])


def _transition_records(concursos: pd.DataFrame) -> Tuple[List[Dict[str, object]], List[Tuple[int, ...]]]:
    if concursos.empty:
        raise ValueError("Base de concursos vazia. Rode primeiro: python main.py --update")
    missing = [col for col in ["concurso", "data_sorteio", *DEZENAS] if col not in concursos.columns]
    if missing:
        raise ValueError(f"Base de concursos sem colunas obrigatorias: {missing}")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    draws = [tuple(_nums_from_row(row)) for _, row in df.iterrows()]
    rows: List[Dict[str, object]] = []
    for idx in range(1, len(df)):
        prev = draws[idx - 1]
        cur = draws[idx]
        prev_set = set(prev)
        cur_set = set(cur)
        repeated = sorted(prev_set & cur_set)
        entered = sorted(cur_set - prev_set)
        exited = sorted(prev_set - cur_set)
        stayed_out = sorted(set(NUMBERS) - prev_set - cur_set)
        repeated_ranges = _range_counts(repeated)
        entered_ranges = _range_counts(entered)
        exited_ranges = _range_counts(exited)
        prev_ranges = _range_counts(prev)
        cur_ranges = _range_counts(cur)
        prev_columns = _column_counts(prev)
        cur_columns = _column_counts(cur)
        deltas_pos = [int(cur[pos] - prev[pos]) for pos in range(PICK_SIZE)]
        rows.append(
            {
                "concurso_origem": int(df.loc[idx - 1, "concurso"]),
                "data_origem": str(df.loc[idx - 1, "data_sorteio"]),
                "concurso_destino": int(df.loc[idx, "concurso"]),
                "data_destino": str(df.loc[idx, "data_sorteio"]),
                "nums_origem": _format_nums(prev),
                "nums_destino": _format_nums(cur),
                "qtd_repetidas": int(len(repeated)),
                "qtd_entraram": int(len(entered)),
                "qtd_sairam": int(len(exited)),
                "qtd_continuaram_fora": int(len(stayed_out)),
                "dezenas_repetidas": _format_nums(repeated) if repeated else "",
                "dezenas_entraram": _format_nums(entered) if entered else "",
                "dezenas_sairam": _format_nums(exited) if exited else "",
                "dezenas_continuaram_fora": _format_nums(stayed_out) if stayed_out else "",
                "soma_origem": int(sum(prev)),
                "soma_destino": int(sum(cur)),
                "delta_soma": int(sum(cur) - sum(prev)),
                "pares_origem": int(sum(1 for n in prev if n % 2 == 0)),
                "pares_destino": int(sum(1 for n in cur if n % 2 == 0)),
                "delta_pares": int(sum(1 for n in cur if n % 2 == 0) - sum(1 for n in prev if n % 2 == 0)),
                "faixas_origem": _signature(prev_ranges),
                "faixas_destino": _signature(cur_ranges),
                "delta_faixas": _signature([cur_ranges[i] - prev_ranges[i] for i in range(5)]),
                "colunas_origem": _signature(prev_columns),
                "colunas_destino": _signature(cur_columns),
                "delta_colunas": _signature([cur_columns[i] - prev_columns[i] for i in range(5)]),
                "repetidas_faixas": _signature(repeated_ranges),
                "entradas_faixas": _signature(entered_ranges),
                "saidas_faixas": _signature(exited_ranges),
                "delta_posicoes": _signature(deltas_pos),
                "delta_pos_medio": round(float(pd.Series(deltas_pos, dtype="float64").mean()), 6),
                "qtd_posicoes_subiram": int(sum(1 for delta in deltas_pos if delta > 0)),
                "qtd_posicoes_desceram": int(sum(1 for delta in deltas_pos if delta < 0)),
                "qtd_posicoes_iguais": int(sum(1 for delta in deltas_pos if delta == 0)),
                "assinatura_transicao": (
                    f"R{len(repeated)}|E{len(entered)}|S{len(exited)}|"
                    f"RF:{_signature(repeated_ranges)}|EF:{_signature(entered_ranges)}|SF:{_signature(exited_ranges)}"
                ),
            }
        )
    return rows, draws


def build_transition_outputs(concursos: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, draws = _transition_records(concursos)
    transitions = pd.DataFrame(rows)
    if transitions.empty:
        return transitions, pd.DataFrame(), pd.DataFrame()

    repeat_values = transitions["qtd_repetidas"].astype(int)
    delta_sums = transitions["delta_soma"].astype(int)
    signature_counts = transitions["assinatura_transicao"].value_counts()
    summary_rows: List[Dict[str, object]] = [
        {"metrica": "pares_consecutivos_analisados", "valor": int(len(transitions)), "observacao": "concurso N comparado com N+1"},
        {"metrica": "media_repetidas", "valor": round(float(repeat_values.mean()), 6), "observacao": ""},
        {"metrica": "mediana_repetidas", "valor": round(float(repeat_values.median()), 6), "observacao": ""},
        {"metrica": "p10_repetidas", "valor": round(float(repeat_values.quantile(0.10)), 6), "observacao": ""},
        {"metrica": "p90_repetidas", "valor": round(float(repeat_values.quantile(0.90)), 6), "observacao": ""},
        {"metrica": "media_delta_soma", "valor": round(float(delta_sums.mean()), 6), "observacao": ""},
        {"metrica": "p10_delta_soma", "valor": round(float(delta_sums.quantile(0.10)), 6), "observacao": ""},
        {"metrica": "p90_delta_soma", "valor": round(float(delta_sums.quantile(0.90)), 6), "observacao": ""},
        {"metrica": "assinatura_transicao_mais_comum", "valor": str(signature_counts.index[0]), "observacao": f"frequencia={int(signature_counts.iloc[0])}"},
    ]
    for repeats, freq in repeat_values.value_counts().sort_index().items():
        summary_rows.append(
            {
                "metrica": f"distribuicao_repetidas_{int(repeats)}",
                "valor": int(freq),
                "observacao": f"{round(float(freq / len(transitions)), 6)} dos pares",
            }
        )

    stats_rows: List[Dict[str, object]] = []
    alpha = 1.0
    for dezena in NUMBERS:
        selected_prev = 0
        omitted_prev = 0
        stayed = 0
        entered = 0
        exited = 0
        stayed_out = 0
        for idx in range(1, len(draws)):
            prev_set = set(draws[idx - 1])
            cur_set = set(draws[idx])
            if dezena in prev_set:
                selected_prev += 1
                if dezena in cur_set:
                    stayed += 1
                else:
                    exited += 1
            else:
                omitted_prev += 1
                if dezena in cur_set:
                    entered += 1
                else:
                    stayed_out += 1
        p_stay = (stayed + alpha) / (selected_prev + 2.0 * alpha)
        p_enter = (entered + alpha) / (omitted_prev + 2.0 * alpha)
        stats_rows.append(
            {
                "dezena": int(dezena),
                "vezes_estava_no_concurso_origem": int(selected_prev),
                "vezes_ficou_no_concurso_seguinte": int(stayed),
                "vezes_saiu_no_concurso_seguinte": int(exited),
                "vezes_estava_fora_no_concurso_origem": int(omitted_prev),
                "vezes_entrou_no_concurso_seguinte": int(entered),
                "vezes_continuou_fora": int(stayed_out),
                "probabilidade_ficar_suavizada": round(float(p_stay), 8),
                "probabilidade_entrar_suavizada": round(float(p_enter), 8),
                "forca_transicao_media": round(float((p_stay + p_enter) / 2.0), 8),
            }
        )
    number_stats = pd.DataFrame(stats_rows)
    return transitions, pd.DataFrame(summary_rows), number_stats


def build_transition_model(concursos: pd.DataFrame) -> TransitionModel:
    transitions, summary, number_stats = build_transition_outputs(concursos)
    if transitions.empty:
        draws = [tuple(_nums_from_row(row)) for _, row in concursos.copy().sort_values("concurso").iterrows()]
        last_draw = draws[-1] if draws else tuple()
        last_set = frozenset(last_draw)
        fallback_scores = {n: 0.60 for n in NUMBERS}
        return TransitionModel(
            last_draw=last_draw,
            last_draw_set=last_set,
            last_sum=sum(last_draw),
            median_repeats=9.0,
            repeats_p10=7.0,
            repeats_p90=11.0,
            delta_sum_p10=-30.0,
            delta_sum_p90=30.0,
            stay_probability=fallback_scores,
            enter_probability=fallback_scores,
            candidate_number_score=fallback_scores,
            full_candidate_number_score=sum(fallback_scores.values()),
            last_range_counts=_range_counts(last_draw),
            not_last_range_counts=_range_counts([n for n in NUMBERS if n not in last_set]),
            common_repeat_signatures=frozenset(),
            common_enter_signatures=frozenset(),
            common_exit_signatures=frozenset(),
            summary={},
        )

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    last_draw = tuple(_nums_from_row(df.iloc[-1]))
    last_set = frozenset(last_draw)
    number_stats = number_stats.set_index("dezena")
    stay_probability = {n: float(number_stats.loc[n, "probabilidade_ficar_suavizada"]) for n in NUMBERS}
    enter_probability = {n: float(number_stats.loc[n, "probabilidade_entrar_suavizada"]) for n in NUMBERS}
    candidate_number_score = {
        n: stay_probability[n] if n in last_set else enter_probability[n]
        for n in NUMBERS
    }
    repeat_values = transitions["qtd_repetidas"].astype(float).tolist()
    delta_sums = transitions["delta_soma"].astype(float).tolist()
    summary_map = {
        "pares_consecutivos_analisados": int(len(transitions)),
        "media_repetidas": round(float(pd.Series(repeat_values).mean()), 6),
        "mediana_repetidas": round(_median(repeat_values, 9.0), 6),
        "p10_repetidas": round(_percentile(repeat_values, 0.10, 7.0), 6),
        "p90_repetidas": round(_percentile(repeat_values, 0.90, 11.0), 6),
        "p10_delta_soma": round(_percentile(delta_sums, 0.10, -30.0), 6),
        "p90_delta_soma": round(_percentile(delta_sums, 0.90, 30.0), 6),
    }
    return TransitionModel(
        last_draw=last_draw,
        last_draw_set=last_set,
        last_sum=sum(last_draw),
        median_repeats=float(summary_map["mediana_repetidas"]),
        repeats_p10=float(summary_map["p10_repetidas"]),
        repeats_p90=float(summary_map["p90_repetidas"]),
        delta_sum_p10=float(summary_map["p10_delta_soma"]),
        delta_sum_p90=float(summary_map["p90_delta_soma"]),
        stay_probability=stay_probability,
        enter_probability=enter_probability,
        candidate_number_score=candidate_number_score,
        full_candidate_number_score=sum(candidate_number_score.values()),
        last_range_counts=_range_counts(last_draw),
        not_last_range_counts=_range_counts([n for n in NUMBERS if n not in last_set]),
        common_repeat_signatures=_top_signatures(Counter(transitions["repetidas_faixas"].astype(str))),
        common_enter_signatures=_top_signatures(Counter(transitions["entradas_faixas"].astype(str))),
        common_exit_signatures=_top_signatures(Counter(transitions["saidas_faixas"].astype(str))),
        summary=summary_map,
    )


def score_transition_from_omitted(omitted: Sequence[int], model: TransitionModel) -> Dict[str, object]:
    omitted_set = set(int(n) for n in omitted)
    repeated_count = PICK_SIZE - sum(1 for n in omitted_set if n in model.last_draw_set)
    score_repetition = _score_0_100(
        _distance_outside_band(repeated_count, model.repeats_p10, model.repeats_p90) * 9.0
        + abs(repeated_count - model.median_repeats) * 2.0
    )

    selected_number_score_sum = model.full_candidate_number_score - sum(model.candidate_number_score.get(int(n), 0.60) for n in omitted_set)
    score_number = round(max(0.0, min(100.0, selected_number_score_sum / PICK_SIZE * 100.0)), 6)

    omitted_last_ranges = [0, 0, 0, 0, 0]
    omitted_not_last_ranges = [0, 0, 0, 0, 0]
    for n in omitted_set:
        idx = _range_index(n)
        if n in model.last_draw_set:
            omitted_last_ranges[idx] += 1
        else:
            omitted_not_last_ranges[idx] += 1

    repeated_ranges = tuple(model.last_range_counts[idx] - omitted_last_ranges[idx] for idx in range(5))
    entered_ranges = tuple(model.not_last_range_counts[idx] - omitted_not_last_ranges[idx] for idx in range(5))
    exited_ranges = tuple(omitted_last_ranges)
    repeat_signature = _signature(repeated_ranges)
    enter_signature = _signature(entered_ranges)
    exit_signature = _signature(exited_ranges)
    selected_sum = FULL_SUM - sum(omitted_set)
    delta_sum = selected_sum - model.last_sum

    structure_score = 45.0
    structure_score += 14.0 if repeat_signature in model.common_repeat_signatures else 0.0
    structure_score += 12.0 if enter_signature in model.common_enter_signatures else 0.0
    structure_score += 12.0 if exit_signature in model.common_exit_signatures else 0.0
    structure_score += 10.0 if model.delta_sum_p10 <= delta_sum <= model.delta_sum_p90 else 0.0
    structure_score += 7.0 if model.repeats_p10 <= repeated_count <= model.repeats_p90 else 0.0
    score_structure = round(max(0.0, min(100.0, structure_score)), 6)

    score_transition = round(
        0.35 * score_repetition + 0.35 * score_number + 0.30 * score_structure,
        6,
    )
    return {
        "score_transicao": score_transition,
        "score_transicao_repeticao": score_repetition,
        "score_transicao_dezenas": score_number,
        "score_transicao_estrutura": score_structure,
        "transicao_repetidas": int(repeated_count),
        "transicao_entraram": int(PICK_SIZE - repeated_count),
        "transicao_sairam": int(PICK_SIZE - repeated_count),
        "transicao_delta_soma": int(delta_sum),
        "transicao_repetidas_faixas": repeat_signature,
        "transicao_entradas_faixas": enter_signature,
        "transicao_saidas_faixas": exit_signature,
    }


def score_transition_candidate(nums: Sequence[int], model: TransitionModel) -> Dict[str, object]:
    selected = set(int(n) for n in nums)
    omitted = [n for n in NUMBERS if n not in selected]
    return score_transition_from_omitted(omitted, model)
