from __future__ import annotations

import heapq
import math
from collections import Counter
from itertools import combinations
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from .backtest_lotofacil import PICK_SIZE
from .context_features import CLIMATE_CONTEXT_COLUMNS, ContextModel, build_context_model, digital_root, score_contextual_candidate
from .normalize import DEZENAS
from .optimizer import (
    MAX_DEZENA,
    VOLANTE_DIAGONAL_1,
    VOLANTE_DIAGONAL_2,
    _common_range_signatures,
    _delays,
    _historical_profile,
    _max_run,
    _pair_counter,
    _range_counts,
    _range_signature,
    _recent_freq,
)
from .transition_analysis import build_transition_model, score_transition_from_omitted


NUMBERS = tuple(range(1, MAX_DEZENA + 1))
EXHAUSTIVE_SOURCE_MODEL = "ensemble_score_v4_exaustivo_transicao"
TOTAL_COMBINATIONS = math.comb(MAX_DEZENA, PICK_SIZE)
OMITTED_SIZE = MAX_DEZENA - PICK_SIZE
PAIR_COUNT = math.comb(PICK_SIZE, 2)
FULL_SUM = sum(NUMBERS)
FULL_EVEN_COUNT = sum(1 for n in NUMBERS if n % 2 == 0)
DEFAULT_EXHAUSTIVE_WEIGHTS: Dict[str, float] = {
    "estatistico": 0.17,
    "historico": 0.115,
    "atraso": 0.055,
    "combinatorio": 0.105,
    "localidade_numerologia": 0.165,
    "climatico": 0.03,
    "cenarios": 0.115,
    "contrarian": 0.085,
    "transicao": 0.105,
    "nao_repeticao_exata": 0.055,
}
WEIGHT_ALIASES = {
    "contextual": "localidade_numerologia",
    "lua_local_numerologia": "localidade_numerologia",
    "localidade_numerologia_lua_contexto": "localidade_numerologia",
    "clima": "climatico",
    "weather": "climatico",
}


def resolve_exhaustive_weights(weights: Mapping[str, float] | None = None) -> Dict[str, float]:
    resolved = dict(DEFAULT_EXHAUSTIVE_WEIGHTS)
    for key, value in (weights or {}).items():
        normalized_key = WEIGHT_ALIASES.get(str(key), str(key))
        if normalized_key not in resolved:
            valid = ", ".join(sorted(resolved))
            raise ValueError(f"Peso desconhecido: {key}. Pesos validos: {valid}.")
        numeric_value = float(value)
        if numeric_value < 0:
            raise ValueError(f"Peso nao pode ser negativo: {key}={value}.")
        resolved[normalized_key] = numeric_value

    total = sum(float(value) for value in resolved.values())
    if total <= 0:
        raise ValueError("A soma dos pesos do score exaustivo precisa ser maior que zero.")
    return {key: round(float(value) / total, 10) for key, value in resolved.items()}


def format_exhaustive_weights(weights: Mapping[str, float]) -> str:
    ordered = [
        "estatistico",
        "historico",
        "atraso",
        "combinatorio",
        "localidade_numerologia",
        "climatico",
        "cenarios",
        "contrarian",
        "transicao",
        "nao_repeticao_exata",
    ]
    labels = {
        "localidade_numerologia": "localidade_numerologia_lua_contexto",
        "climatico": "clima_temperatura_umidade_pressao_chuva",
    }
    return ";".join(f"{labels.get(key, key)}={float(weights[key]):.4f}" for key in ordered)


def _nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _line_counts_from_ranges(ranges: Sequence[int]) -> List[int]:
    return [int(value) for value in ranges]


def _column_counts_from_omitted(omitted: Sequence[int]) -> List[int]:
    omitted_counts = [0, 0, 0, 0, 0]
    for n in omitted:
        omitted_counts[(int(n) - 1) % 5] += 1
    return [5 - value for value in omitted_counts]


def _max_run_from_omitted(omitted_set: set[int]) -> int:
    best = cur = 0
    for n in NUMBERS:
        if n in omitted_set:
            cur = 0
            continue
        cur += 1
        best = max(best, cur)
    return int(best)


def _diagonal_strength_from_omitted(omitted_set: set[int]) -> int:
    selected_diag_1 = len(VOLANTE_DIAGONAL_1 - omitted_set)
    selected_diag_2 = len(VOLANTE_DIAGONAL_2 - omitted_set)
    return max(selected_diag_1, selected_diag_2)


def _score_0_100(penalty: float) -> float:
    return round(max(0.0, min(100.0, 100.0 - float(penalty))), 6)


def _distance_outside_band(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _target_context_keys(model: ContextModel) -> List[Tuple[str, float]]:
    target = model.target
    keys: List[Tuple[str, float]] = [
        (f"weekday:{target.dia_semana_numero}", 0.13),
        (f"month:{target.mes}", 0.08),
        (f"quarter:{target.trimestre}", 0.06),
        (f"semester:{target.semestre}", 0.05),
        (f"season:{target.estacao_do_ano}", 0.10),
        (f"moon:{target.fase_lua}", 0.16),
        (f"numerology_date:{target.numerologia_data_raiz}", 0.10),
        (f"numerology_concurso:{target.numerologia_concurso_raiz}", 0.09),
        (f"numerology_day_month:{target.numerologia_dia_mes_raiz}", 0.06),
    ]
    if target.local_sorteio_assumido:
        keys.append((f"local:{target.local_sorteio_assumido}", 0.06))
    if target.cidade_sorteio_assumida:
        keys.append((f"cidade:{target.cidade_sorteio_assumida}", 0.04))
    if target.uf_sorteio_assumida:
        keys.append((f"uf:{target.uf_sorteio_assumida}", 0.03))
    if target.bairro_sorteio_assumido:
        keys.append((f"bairro:{target.bairro_sorteio_assumido}", 0.03))
    if target.cidade_sorteio_assumida and target.uf_sorteio_assumida:
        keys.append((f"cidade_uf:{target.cidade_sorteio_assumida}|{target.uf_sorteio_assumida}", 0.04))
    return keys


def _context_number_scores(model: ContextModel) -> Dict[int, float]:
    target = model.target
    keys = _target_context_keys(model)
    total_weight = sum(weight for _key, weight in keys)
    scores = {n: 50.0 for n in NUMBERS}
    if total_weight <= 0:
        return scores

    weighted = {n: 0.0 for n in NUMBERS}
    for key, weight in keys:
        sample_size = int(model.sample_sizes.get(key, 0))
        counter = model.counters.get(key, Counter())
        if sample_size <= 0:
            for n in NUMBERS:
                weighted[n] += weight * 50.0
            continue
        expected = sample_size * PICK_SIZE / MAX_DEZENA
        shrink = min(1.0, sample_size / 50.0)
        for n in NUMBERS:
            raw = 50.0 + (float(counter.get(n, 0)) - expected) * 7.0
            weighted[n] += weight * max(0.0, min(100.0, 50.0 + (raw - 50.0) * shrink))

    roots = {
        target.numerologia_data_raiz,
        target.numerologia_concurso_raiz,
        target.numerologia_dia_mes_raiz,
        digital_root(target.dia_semana_numero),
        digital_root(target.mes),
    }
    for n in NUMBERS:
        numerology_bonus = 8.0 if digital_root(n) in roots else -3.0
        scores[n] = round(max(0.0, min(100.0, weighted[n] / total_weight + numerology_bonus)), 6)
    return scores


def _target_climate_keys(model: ContextModel) -> List[Tuple[str, float]]:
    target = model.target
    column_weights = {
        "clima_temperatura_faixa": 0.18,
        "clima_sensacao_faixa": 0.14,
        "clima_umidade_faixa": 0.16,
        "clima_pressao_faixa": 0.14,
        "clima_chuva_faixa": 0.14,
        "clima_anomalia_faixa": 0.16,
        "clima_assinatura": 0.08,
    }
    keys: List[Tuple[str, float]] = []
    for column, prefix in CLIMATE_CONTEXT_COLUMNS:
        value = getattr(target, column, "indisponivel")
        if value and str(value) != "indisponivel":
            keys.append((f"{prefix}:{value}", column_weights.get(column, 0.10)))
    return keys


def _climate_number_scores(model: ContextModel) -> Dict[int, float]:
    keys = _target_climate_keys(model)
    total_weight = sum(weight for _key, weight in keys)
    scores = {n: 50.0 for n in NUMBERS}
    if total_weight <= 0:
        return scores

    weighted = {n: 0.0 for n in NUMBERS}
    for key, weight in keys:
        sample_size = int(model.sample_sizes.get(key, 0))
        counter = model.counters.get(key, Counter())
        if sample_size <= 0:
            for n in NUMBERS:
                weighted[n] += weight * 50.0
            continue
        expected = sample_size * PICK_SIZE / MAX_DEZENA
        shrink = min(1.0, sample_size / 40.0)
        for n in NUMBERS:
            raw = 50.0 + (float(counter.get(n, 0)) - expected) * 6.0
            weighted[n] += weight * max(0.0, min(100.0, 50.0 + (raw - 50.0) * shrink))

    for n in NUMBERS:
        scores[n] = round(max(0.0, min(100.0, weighted[n] / total_weight)), 6)
    return scores


def _pair_selected_sum_from_omitted(
    *,
    total_pair_sum: float,
    incident_pair_sum: Dict[int, float],
    pair_matrix: List[List[float]],
    omitted: Sequence[int],
) -> float:
    incident = sum(float(incident_pair_sum.get(int(n), 0.0)) for n in omitted)
    omitted_pair_sum = 0.0
    for left_idx in range(len(omitted)):
        left = int(omitted[left_idx])
        for right in omitted[left_idx + 1 :]:
            omitted_pair_sum += pair_matrix[left][int(right)]
    return float(total_pair_sum - incident + omitted_pair_sum)


def _scenario_score(
    *,
    total: int,
    max_run: int,
    ranges: Sequence[int],
    columns: Sequence[int],
    diagonal_strength: int,
    profile: Dict[str, float],
    common_signatures: set[str],
    selected: Sequence[int],
) -> float:
    signature = "-".join(str(value) for value in ranges)
    scenario_hits = 0
    scenario_hits += int(profile["sum_p10"] <= total <= profile["sum_p90"])
    scenario_hits += int(signature in common_signatures)
    scenario_hits += int(max_run >= 5)
    scenario_hits += int(any(value == 5 for value in ranges) or any(value == 5 for value in columns) or diagonal_strength >= 4)
    scenario_hits += int(sum(1 for n in selected if n in {1, 5, 13, 21, 22, 25}) >= 2)
    return round(min(100.0, 45.0 + scenario_hits * 11.0), 6)


def _contrarian_score(selected: Sequence[int], *, max_run: int, ranges: Sequence[int], columns: Sequence[int]) -> float:
    selected_set = set(int(n) for n in selected)
    score = 45.0
    score += 10.0 if 1 in selected_set else 0.0
    score += 8.0 if 13 in selected_set else 0.0
    score += 6.0 if 22 in selected_set else 0.0
    score += min(14.0, len(selected_set & {1, 5, 21, 25}) * 3.5)
    score += 8.0 if max_run >= 5 else 0.0
    score += 7.0 if any(value == 5 for value in ranges) or any(value == 5 for value in columns) else 0.0
    return round(max(0.0, min(100.0, score)), 6)


def _ranked_heap_rows(heap: List[Tuple[float, int, Dict[str, object]]]) -> pd.DataFrame:
    rows = [row for _score, _idx, row in sorted(heap, key=lambda item: (-item[0], item[1]))]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def build_exhaustive_candidates(
    concursos: pd.DataFrame,
    *,
    top_games: int = 5000,
    draw_hour: int = 20,
    draw_minute: int = 0,
    limit_combinations: int | None = None,
    weights: Mapping[str, float] | None = None,
    climate_features: pd.DataFrame | None = None,
    target_climate: Mapping[str, object] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if concursos.empty:
        raise ValueError("Base de concursos vazia. Rode primeiro: python main.py --update")

    resolved_weights = resolve_exhaustive_weights(weights)
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
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
        target_climate=target_climate,
    )
    context_scores = _context_number_scores(context_model)
    climate_scores = _climate_number_scores(context_model)
    transition_model = build_transition_model(df)

    recent_total = sum(float(recent_freq.get(n, 0)) for n in NUMBERS)
    delay_total = sum(float(delays.get(n, 0)) for n in NUMBERS)
    context_total = sum(float(context_scores.get(n, 50.0)) for n in NUMBERS)
    climate_total = sum(float(climate_scores.get(n, 50.0)) for n in NUMBERS)
    pair_values = list(pair_freq.values())
    pair_median = float(pd.Series(pair_values).median()) if pair_values else 0.0
    pair_matrix = [[0.0 for _ in range(MAX_DEZENA + 1)] for _ in range(MAX_DEZENA + 1)]
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

    keep_top = max(2, int(top_games))
    heap: List[Tuple[float, int, Dict[str, object]]] = []
    evaluated = 0
    exact_historical_seen = 0

    for idx, omitted in enumerate(combinations(NUMBERS, OMITTED_SIZE), start=1):
        if limit_combinations is not None and evaluated >= int(limit_combinations):
            break
        evaluated += 1
        omitted_set = set(int(n) for n in omitted)
        selected = tuple(n for n in NUMBERS if n not in omitted_set)
        selected_sum = FULL_SUM - sum(omitted)
        selected_pairs = FULL_EVEN_COUNT - sum(1 for n in omitted if n % 2 == 0)
        ranges = [5, 5, 5, 5, 5]
        for n in omitted:
            ranges[(int(n) - 1) // 5] -= 1
        columns = _column_counts_from_omitted(omitted)
        max_run = _max_run_from_omitted(omitted_set)
        overlap_last = PICK_SIZE - sum(1 for n in omitted if int(n) in last_draw)
        diagonal_strength = _diagonal_strength_from_omitted(omitted_set)
        exact_historical = tuple(selected) in existing_draws
        exact_historical_seen += int(exact_historical)

        signature = "-".join(str(value) for value in ranges)
        stat_penalty = 0.0
        stat_penalty += _distance_outside_band(selected_sum, profile["sum_p10"], profile["sum_p90"]) / 1.6
        stat_penalty += _distance_outside_band(selected_pairs, profile["pairs_p10"], profile["pairs_p90"]) * 3.0
        stat_penalty += 0.0 if signature in common_signatures else sum(abs(value - 3) for value in ranges) * 1.8
        stat_penalty += abs(overlap_last - profile["median_overlap"]) * 2.2
        stat_penalty += max(0.0, max_run - profile["run_p95"]) * 2.2
        score_estatistico = _score_0_100(stat_penalty)

        selected_recent_sum = recent_total - sum(float(recent_freq.get(int(n), 0)) for n in omitted)
        avg_recent = selected_recent_sum / PICK_SIZE
        expected_recent = 100 * PICK_SIZE / MAX_DEZENA
        score_historico = _score_0_100(abs(avg_recent - expected_recent) * 1.2)

        selected_delay_sum = delay_total - sum(float(delays.get(int(n), 0)) for n in omitted)
        avg_delay = selected_delay_sum / PICK_SIZE
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
        score_contextual = round(max(0.0, min(100.0, selected_context_sum / PICK_SIZE)), 6)
        selected_climate_sum = climate_total - sum(float(climate_scores.get(int(n), 50.0)) for n in omitted)
        score_climatico = round(max(0.0, min(100.0, selected_climate_sum / PICK_SIZE)), 6)
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
        score_localidade_numerologia = score_contextual
        transition_detail = score_transition_from_omitted(omitted, transition_model)
        score_transicao = float(transition_detail["score_transicao"])

        score_final = round(
            resolved_weights["estatistico"] * score_estatistico
            + resolved_weights["historico"] * score_historico
            + resolved_weights["atraso"] * score_atraso
            + resolved_weights["combinatorio"] * score_combinatorio
            + resolved_weights["localidade_numerologia"] * score_localidade_numerologia
            + resolved_weights["climatico"] * score_climatico
            + resolved_weights["cenarios"] * score_cenarios
            + resolved_weights["contrarian"] * score_contrarian
            + resolved_weights["transicao"] * score_transicao
            + resolved_weights["nao_repeticao_exata"] * (100.0 if not exact_historical else 92.0),
            6,
        )

        if len(heap) < keep_top or score_final > heap[0][0]:
            row: Dict[str, object] = {
                "nums": _format_nums(selected),
                "source_model": EXHAUSTIVE_SOURCE_MODEL,
                "metodo": EXHAUSTIVE_SOURCE_MODEL,
                "score_final": score_final,
                "score_estatistico": score_estatistico,
                "score_historico": score_historico,
                "score_atraso": score_atraso,
                "score_combinatorio": score_combinatorio,
                "score_contextual": score_contextual,
                "score_localidade_numerologia": score_localidade_numerologia,
                "score_climatico": score_climatico,
                "score_cenarios": score_cenarios,
                "score_contrarian": score_contrarian,
                "score_transicao": score_transicao,
                "score_transicao_repeticao": float(transition_detail["score_transicao_repeticao"]),
                "score_transicao_dezenas": float(transition_detail["score_transicao_dezenas"]),
                "score_transicao_estrutura": float(transition_detail["score_transicao_estrutura"]),
                "soma": int(selected_sum),
                "qtd_pares": int(selected_pairs),
                "overlap_ultimo": int(overlap_last),
                "transicao_repetidas": int(transition_detail["transicao_repetidas"]),
                "transicao_entraram": int(transition_detail["transicao_entraram"]),
                "transicao_sairam": int(transition_detail["transicao_sairam"]),
                "transicao_delta_soma": int(transition_detail["transicao_delta_soma"]),
                "transicao_repetidas_faixas": str(transition_detail["transicao_repetidas_faixas"]),
                "transicao_entradas_faixas": str(transition_detail["transicao_entradas_faixas"]),
                "transicao_saidas_faixas": str(transition_detail["transicao_saidas_faixas"]),
                "maior_sequencia": int(max_run),
                "faixas_5": " ".join(str(v) for v in ranges),
                "linhas": " ".join(str(v) for v in _line_counts_from_ranges(ranges)),
                "colunas": " ".join(str(v) for v in columns),
                "assinatura_faixas": signature,
                "diagonal_strength": int(diagonal_strength),
                "media_freq_recente": round(float(avg_recent), 6),
                "media_atraso": round(float(avg_delay), 6),
                "media_freq_pares": round(float(avg_pair_freq), 6),
                "ja_saiu_exatamente_no_historico": int(exact_historical),
                "total_combinacoes_avaliadas": int(TOTAL_COMBINATIONS if limit_combinations is None else evaluated),
                "concurso_base_inicial": int(df["concurso"].min()),
                "concurso_base_final": int(df["concurso"].max()),
                "contexto_clima_status": context_model.target.clima_status,
                "contexto_clima_fonte": context_model.target.clima_fonte,
                "contexto_clima_temperature_2m": context_model.target.clima_temperature_2m,
                "contexto_clima_apparent_temperature": context_model.target.clima_apparent_temperature,
                "contexto_clima_relative_humidity_2m": context_model.target.clima_relative_humidity_2m,
                "contexto_clima_surface_pressure": context_model.target.clima_surface_pressure,
                "contexto_clima_precipitation": context_model.target.clima_precipitation,
                "contexto_clima_temperature_media_30d": context_model.target.clima_temperature_media_30d,
                "contexto_clima_temperature_anomalia": context_model.target.clima_temperature_anomalia,
                "contexto_clima_temperatura_faixa": context_model.target.clima_temperatura_faixa,
                "contexto_clima_sensacao_faixa": context_model.target.clima_sensacao_faixa,
                "contexto_clima_umidade_faixa": context_model.target.clima_umidade_faixa,
                "contexto_clima_pressao_faixa": context_model.target.clima_pressao_faixa,
                "contexto_clima_chuva_faixa": context_model.target.clima_chuva_faixa,
                "contexto_clima_anomalia_faixa": context_model.target.clima_anomalia_faixa,
                "contexto_clima_assinatura": context_model.target.clima_assinatura,
            }
            item = (score_final, idx, row)
            if len(heap) < keep_top:
                heapq.heappush(heap, item)
            else:
                heapq.heapreplace(heap, item)

    candidates = _ranked_heap_rows(heap)
    if not candidates.empty:
        candidates["total_combinacoes_avaliadas"] = int(evaluated)
        detail_rows = []
        for _, row in candidates.iterrows():
            nums = [int(part) for part in str(row["nums"]).split()]
            detail_rows.append(score_contextual_candidate(nums, context_model))
        detail_df = pd.DataFrame(detail_rows).add_prefix("detalhe_")
        candidates = pd.concat([candidates.reset_index(drop=True), detail_df.reset_index(drop=True)], axis=1)
        if "detalhe_score_contextual" in candidates.columns:
            candidates["score_contextual_detalhado"] = candidates["detalhe_score_contextual"]
        for col in list(candidates.columns):
            if col.startswith("detalhe_contexto_"):
                candidates[col.replace("detalhe_", "", 1)] = candidates[col]
    summary = pd.DataFrame(
        [
            {"metrica": "source_model", "valor": EXHAUSTIVE_SOURCE_MODEL},
            {"metrica": "combinacoes_possiveis", "valor": int(TOTAL_COMBINATIONS)},
            {"metrica": "combinacoes_avaliadas", "valor": int(evaluated)},
            {"metrica": "top_games_salvos", "valor": int(len(candidates))},
            {"metrica": "historico_concursos_usados", "valor": int(len(df))},
            {"metrica": "primeiro_concurso_base", "valor": int(df["concurso"].min())},
            {"metrica": "ultimo_concurso_base", "valor": int(df["concurso"].max())},
            {"metrica": "combos_top_que_ja_sairam", "valor": int(candidates["ja_saiu_exatamente_no_historico"].sum()) if not candidates.empty else 0},
            {"metrica": "combos_historicos_encontrados_durante_varredura", "valor": int(exact_historical_seen)},
            {"metrica": "draw_hour_brasilia", "valor": int(draw_hour)},
            {"metrica": "draw_minute_brasilia", "valor": int(draw_minute)},
            {"metrica": "data_proximo_concurso", "valor": context_model.target.data_proximo_concurso},
            {"metrica": "dia_semana_proximo_concurso", "valor": context_model.target.dia_semana_nome},
            {"metrica": "fase_lua_proximo_concurso", "valor": context_model.target.fase_lua},
            {"metrica": "idade_lua_proximo_concurso", "valor": context_model.target.idade_lua},
            {"metrica": "iluminacao_lua_proximo_concurso", "valor": context_model.target.iluminacao_lua_percentual},
            {"metrica": "numerologia_data_raiz", "valor": context_model.target.numerologia_data_raiz},
            {"metrica": "numerologia_concurso_raiz", "valor": context_model.target.numerologia_concurso_raiz},
            {"metrica": "numerologia_dia_mes_raiz", "valor": context_model.target.numerologia_dia_mes_raiz},
            {"metrica": "local_sorteio_assumido", "valor": context_model.target.local_sorteio_assumido},
            {"metrica": "cidade_sorteio_assumida", "valor": context_model.target.cidade_sorteio_assumida},
            {"metrica": "uf_sorteio_assumida", "valor": context_model.target.uf_sorteio_assumida},
            {"metrica": "bairro_sorteio_assumido", "valor": context_model.target.bairro_sorteio_assumido or "indisponivel_na_base"},
            {"metrica": "observacao_localidade", "valor": context_model.target.observacao_localidade},
            {"metrica": "clima_status", "valor": context_model.target.clima_status},
            {"metrica": "clima_fonte", "valor": context_model.target.clima_fonte},
            {"metrica": "clima_temperature_2m", "valor": context_model.target.clima_temperature_2m},
            {"metrica": "clima_apparent_temperature", "valor": context_model.target.clima_apparent_temperature},
            {"metrica": "clima_relative_humidity_2m", "valor": context_model.target.clima_relative_humidity_2m},
            {"metrica": "clima_surface_pressure", "valor": context_model.target.clima_surface_pressure},
            {"metrica": "clima_precipitation", "valor": context_model.target.clima_precipitation},
            {"metrica": "clima_temperature_media_30d", "valor": context_model.target.clima_temperature_media_30d},
            {"metrica": "clima_temperature_anomalia", "valor": context_model.target.clima_temperature_anomalia},
            {"metrica": "clima_temperatura_faixa", "valor": context_model.target.clima_temperatura_faixa},
            {"metrica": "clima_sensacao_faixa", "valor": context_model.target.clima_sensacao_faixa},
            {"metrica": "clima_umidade_faixa", "valor": context_model.target.clima_umidade_faixa},
            {"metrica": "clima_pressao_faixa", "valor": context_model.target.clima_pressao_faixa},
            {"metrica": "clima_chuva_faixa", "valor": context_model.target.clima_chuva_faixa},
            {"metrica": "clima_anomalia_faixa", "valor": context_model.target.clima_anomalia_faixa},
            {"metrica": "clima_assinatura", "valor": context_model.target.clima_assinatura},
            {"metrica": "transicao_pares_consecutivos_analisados", "valor": transition_model.summary.get("pares_consecutivos_analisados", 0)},
            {"metrica": "transicao_media_repetidas", "valor": transition_model.summary.get("media_repetidas", "")},
            {"metrica": "transicao_mediana_repetidas", "valor": transition_model.summary.get("mediana_repetidas", "")},
            {"metrica": "transicao_p10_repetidas", "valor": transition_model.summary.get("p10_repetidas", "")},
            {"metrica": "transicao_p90_repetidas", "valor": transition_model.summary.get("p90_repetidas", "")},
            {"metrica": "score_weights", "valor": format_exhaustive_weights(resolved_weights)},
        ]
    )
    return candidates, summary
