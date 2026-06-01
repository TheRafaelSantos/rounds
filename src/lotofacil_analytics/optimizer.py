from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .backtest_lotofacil import PICK_SIZE
from .context_features import ContextModel, build_context_model, score_contextual_candidate
from .normalize import DEZENAS


MAX_DEZENA = 25
VOLANTE_DIAGONAL_1 = {1, 7, 13, 19, 25}
VOLANTE_DIAGONAL_2 = {5, 9, 13, 17, 21}


@dataclass(frozen=True)
class OptimizerSummary:
    candidates_rows: int
    summary_rows: int
    candidates_csv_path: str
    summary_csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Fase 8",
                "Acao: optimize",
                f"Candidatos ranqueados: {self.candidates_rows}",
                f"Linhas resumo: {self.summary_rows}",
                f"CSV candidatos: {self.candidates_csv_path}",
                f"CSV resumo: {self.summary_csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Otimizacao gerada para apoiar a selecao final.",
            ]
        )


def _nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _range_counts(nums: Sequence[int]) -> List[int]:
    return [sum(1 for n in nums if start <= int(n) <= start + 4) for start in [1, 6, 11, 16, 21]]


def _line_counts(nums: Sequence[int]) -> List[int]:
    out = [0, 0, 0, 0, 0]
    for n in nums:
        out[(int(n) - 1) // 5] += 1
    return out


def _column_counts(nums: Sequence[int]) -> List[int]:
    out = [0, 0, 0, 0, 0]
    for n in nums:
        out[(int(n) - 1) % 5] += 1
    return out


def _max_run(nums: Sequence[int]) -> int:
    ordered = sorted(int(n) for n in nums)
    best = cur = 1
    for idx in range(1, len(ordered)):
        if ordered[idx] == ordered[idx - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _has_full_line_or_column(nums: Sequence[int]) -> bool:
    return any(value == 5 for value in _line_counts(nums)) or any(value == 5 for value in _column_counts(nums))


def _diagonal_strength(nums: Sequence[int]) -> int:
    nums_set = set(int(n) for n in nums)
    return max(len(nums_set & VOLANTE_DIAGONAL_1), len(nums_set & VOLANTE_DIAGONAL_2))


def _score_0_100(penalty: float) -> float:
    return round(max(0.0, min(100.0, 100.0 - float(penalty))), 6)


def _historical_profile(draws: Sequence[Sequence[int]]) -> Dict[str, float]:
    sums = [sum(draw) for draw in draws]
    pairs = [sum(1 for n in draw if n % 2 == 0) for draw in draws]
    runs = [_max_run(draw) for draw in draws]
    overlaps: List[int] = []
    previous: set[int] | None = None
    for draw in draws:
        cur = set(draw)
        if previous is not None:
            overlaps.append(len(cur & previous))
        previous = cur
    sum_series = pd.Series(sums)
    pair_series = pd.Series(pairs)
    run_series = pd.Series(runs)
    profile = {
        "median_sum": float(pd.Series(sums).median()),
        "sum_p10": float(sum_series.quantile(0.10)),
        "sum_p25": float(sum_series.quantile(0.25)),
        "sum_p75": float(sum_series.quantile(0.75)),
        "sum_p90": float(sum_series.quantile(0.90)),
        "median_pairs": float(pd.Series(pairs).median()),
        "pairs_p10": float(pair_series.quantile(0.10)),
        "pairs_p90": float(pair_series.quantile(0.90)),
        "median_overlap": float(pd.Series(overlaps).median()) if overlaps else 9.0,
        "run_p80": float(run_series.quantile(0.80)),
        "run_p95": float(run_series.quantile(0.95)),
    }
    return profile


def _recent_freq(draws: Sequence[Sequence[int]], window: int = 100) -> Counter[int]:
    counter: Counter[int] = Counter()
    for draw in draws[-window:]:
        counter.update(int(n) for n in draw)
    return counter


def _pair_counter(draws: Sequence[Sequence[int]]) -> Counter[Tuple[int, int]]:
    counter: Counter[Tuple[int, int]] = Counter()
    for draw in draws:
        counter.update(tuple(combo) for combo in combinations(sorted(draw), 2))
    return counter


def _range_signature(nums: Sequence[int]) -> str:
    return "-".join(str(value) for value in _range_counts(nums))


def _common_range_signatures(draws: Sequence[Sequence[int]], *, top_n: int = 30) -> set[str]:
    counter = Counter(_range_signature(draw) for draw in draws)
    return set(sig for sig, _count in counter.most_common(top_n))


def _delays(draws: Sequence[Sequence[int]]) -> Dict[int, int]:
    last_seen = {n: None for n in range(1, MAX_DEZENA + 1)}
    for idx, draw in enumerate(draws):
        for n in draw:
            last_seen[int(n)] = idx
    total = len(draws)
    return {n: total if last_seen[n] is None else total - int(last_seen[n]) - 1 for n in range(1, MAX_DEZENA + 1)}


def _distance_outside_band(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _scenario_score(nums: Sequence[int], *, profile: Dict[str, float], common_signatures: set[str]) -> float:
    total = sum(int(n) for n in nums)
    max_run = _max_run(nums)
    signature = _range_signature(nums)
    high_band = profile["sum_p25"] <= total <= profile["sum_p75"]
    low_band = profile["sum_p10"] <= total < profile["sum_p25"]
    high_sum_band = profile["sum_p75"] < total <= profile["sum_p90"]
    scenario_hits = 0
    scenario_hits += int(high_band or low_band or high_sum_band)
    scenario_hits += int(signature in common_signatures)
    scenario_hits += int(max_run >= 5)
    scenario_hits += int(_has_full_line_or_column(nums) or _diagonal_strength(nums) >= 4)
    scenario_hits += int(sum(1 for n in nums if n in {1, 5, 13, 21, 22, 25}) >= 2)
    return round(min(100.0, 45.0 + scenario_hits * 11.0), 6)


def _contrarian_score(nums: Sequence[int]) -> float:
    nums_set = set(int(n) for n in nums)
    score = 45.0
    score += 10.0 if 1 in nums_set else 0.0
    score += 8.0 if 13 in nums_set else 0.0
    score += 6.0 if 22 in nums_set else 0.0
    score += min(14.0, len(nums_set & {1, 5, 21, 25}) * 3.5)
    score += 8.0 if _max_run(nums) >= 5 else 0.0
    score += 7.0 if _has_full_line_or_column(nums) else 0.0
    return round(max(0.0, min(100.0, score)), 6)


def _weighted_candidate(rng: random.Random, freq_recent: Counter[int], delays: Dict[int, int]) -> List[int]:
    selected: set[int] = set()
    delay_series = pd.Series(list(delays.values()), dtype="float64")
    median_delay = float(delay_series.median()) if len(delay_series) else 0.0
    while len(selected) < PICK_SIZE:
        available = [n for n in range(1, MAX_DEZENA + 1) if n not in selected]
        weights = []
        for n in available:
            recent_component = 1.0 + (float(freq_recent.get(n, 0)) / 100.0)
            delay_component = 1.0 + min(0.50, max(0.0, float(delays.get(n, 0)) - median_delay) / 20.0)
            contrarian_component = 1.12 if n in {1, 13, 22} else 1.0
            weights.append(recent_component * delay_component * contrarian_component)
        selected.add(rng.choices(available, weights=weights, k=1)[0])
    return sorted(selected)


def _candidate_matches_profile(nums: Sequence[int], profile_name: str, profile: Dict[str, float], common_signatures: set[str]) -> bool:
    total = sum(int(n) for n in nums)
    if profile_name == "soma_baixa":
        return profile["sum_p10"] <= total <= profile["sum_p25"]
    if profile_name == "soma_media":
        return profile["sum_p25"] <= total <= profile["sum_p75"]
    if profile_name == "soma_alta":
        return profile["sum_p75"] <= total <= profile["sum_p90"]
    if profile_name == "sequencia_forte":
        return _max_run(nums) >= 5
    if profile_name == "visual_forte":
        return _has_full_line_or_column(nums) or _diagonal_strength(nums) >= 4
    if profile_name == "contrarian_controlado":
        return len(set(nums) & {1, 13, 22, 5, 21, 25}) >= 2
    if profile_name == "faixa_alta_reforcada":
        return sum(1 for n in nums if 21 <= int(n) <= 25) >= 3
    if profile_name == "assinatura_historica":
        return _range_signature(nums) in common_signatures
    return True


def _profiled_candidate(
    rng: random.Random,
    *,
    profile_name: str,
    profile: Dict[str, float],
    common_signatures: set[str],
    freq_recent: Counter[int],
    delays: Dict[int, int],
) -> List[int]:
    for _attempt in range(250):
        nums = _weighted_candidate(rng, freq_recent, delays) if rng.random() < 0.65 else _random_candidate(rng)
        if _candidate_matches_profile(nums, profile_name, profile, common_signatures):
            return nums
    return _weighted_candidate(rng, freq_recent, delays)


def score_candidate(
    nums: Sequence[int],
    *,
    profile: Dict[str, float],
    last_draw: Sequence[int],
    freq_recent: Counter[int],
    pair_freq: Counter[Tuple[int, int]],
    context_model: ContextModel | None = None,
    common_signatures: set[str] | None = None,
) -> Dict[str, float | int | str]:
    nums = sorted(int(n) for n in nums)
    pairs = sum(1 for n in nums if n % 2 == 0)
    total = sum(nums)
    ranges = _range_counts(nums)
    lines = _line_counts(nums)
    columns = _column_counts(nums)
    overlap_last = len(set(nums) & set(last_draw))
    max_run = _max_run(nums)

    common_signatures = common_signatures or set()
    stat_penalty = 0.0
    stat_penalty += _distance_outside_band(total, profile["sum_p10"], profile["sum_p90"]) / 1.6
    stat_penalty += _distance_outside_band(pairs, profile["pairs_p10"], profile["pairs_p90"]) * 3.0
    range_penalty = 0.0 if _range_signature(nums) in common_signatures else sum(abs(value - 3) for value in ranges) * 1.8
    stat_penalty += range_penalty
    stat_penalty += abs(overlap_last - profile["median_overlap"]) * 2.2
    stat_penalty += max(0.0, max_run - profile["run_p95"]) * 2.2
    score_estatistico = _score_0_100(stat_penalty)

    avg_recent_freq = sum(freq_recent.get(n, 0) for n in nums) / len(nums)
    expected_recent_freq = 100 * PICK_SIZE / MAX_DEZENA
    historico_penalty = abs(avg_recent_freq - expected_recent_freq) * 1.2
    score_historico = _score_0_100(historico_penalty)

    popularity_penalty = 0.0
    popularity_penalty += max(0, sum(1 for n in nums if n <= 12) - 10) * 3.0
    popularity_penalty += max(0, sum(1 for n in nums if n <= 15) - 12) * 2.2
    popularity_penalty += sum(4.0 for value in lines if value == 5)
    popularity_penalty += sum(3.5 for value in columns if value == 5)
    popularity_penalty += max(0, len(set(nums) & VOLANTE_DIAGONAL_1) - 4) * 4.0
    popularity_penalty += max(0, len(set(nums) & VOLANTE_DIAGONAL_2) - 4) * 4.0
    popularity_penalty += max(0.0, max_run - profile["run_p95"]) * 2.0
    score_anti_popularidade = _score_0_100(popularity_penalty)

    pair_values = [pair_freq.get(tuple(combo), 0) for combo in combinations(nums, 2)]
    avg_pair_freq = sum(pair_values) / len(pair_values)
    pair_median = float(pd.Series(list(pair_freq.values())).median()) if pair_freq else 0.0
    combinatorio_penalty = max(0.0, avg_pair_freq - pair_median) / 12.0
    score_combinatorio = _score_0_100(combinatorio_penalty)
    context_row: Dict[str, object] = (
        score_contextual_candidate(nums, context_model)
        if context_model is not None
        else {"score_contextual": 50.0}
    )
    score_contextual = float(context_row["score_contextual"])
    score_cenarios = _scenario_score(nums, profile=profile, common_signatures=common_signatures)
    score_contrarian = _contrarian_score(nums)

    score_final = round(
        0.27 * score_estatistico
        + 0.18 * score_historico
        + 0.12 * score_anti_popularidade
        + 0.12 * score_combinatorio
        + 0.15 * score_contextual
        + 0.10 * score_cenarios
        + 0.06 * score_contrarian,
        6,
    )

    result: Dict[str, float | int | str] = {
        "nums": _format_nums(nums),
        "score_final": score_final,
        "score_estatistico": score_estatistico,
        "score_historico": score_historico,
        "score_anti_popularidade": score_anti_popularidade,
        "score_combinatorio": score_combinatorio,
        "score_contextual": round(float(score_contextual), 6),
        "score_cenarios": score_cenarios,
        "score_contrarian": score_contrarian,
        "soma": int(total),
        "qtd_pares": int(pairs),
        "overlap_ultimo": int(overlap_last),
        "maior_sequencia": int(max_run),
        "faixas_5": " ".join(str(v) for v in ranges),
        "linhas": " ".join(str(v) for v in lines),
        "colunas": " ".join(str(v) for v in columns),
        "media_freq_recente": round(float(avg_recent_freq), 6),
        "media_freq_pares": round(float(avg_pair_freq), 6),
    }
    for key, value in context_row.items():
        if key != "score_contextual":
            result[key] = value  # type: ignore[assignment]
    return result


def _random_candidate(rng: random.Random) -> List[int]:
    return sorted(rng.sample(range(1, MAX_DEZENA + 1), PICK_SIZE))


def _child_from_parents(rng: random.Random, parent_a: Sequence[int], parent_b: Sequence[int]) -> List[int]:
    selected = set(rng.sample(list(parent_a), 8))
    for n in rng.sample(list(parent_b), len(parent_b)):
        if len(selected) >= PICK_SIZE:
            break
        selected.add(int(n))
    while len(selected) < PICK_SIZE:
        selected.add(rng.randint(1, MAX_DEZENA))
    if rng.random() < 0.35:
        remove = rng.choice(sorted(selected))
        selected.remove(remove)
        while len(selected) < PICK_SIZE:
            selected.add(rng.randint(1, MAX_DEZENA))
    return sorted(selected)


def build_optimized_candidates(
    concursos: pd.DataFrame,
    *,
    seed: int = 123,
    candidate_pool: int = 10000,
    top_games: int = 100,
    generations: int = 20,
    population: int = 80,
    draw_hour: int = 20,
    draw_minute: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if concursos.empty:
        raise ValueError("Base de concursos vazia. Rode primeiro: python main.py --update")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    draws = [_nums_from_row(row) for _, row in df.iterrows()]
    existing = {tuple(draw) for draw in draws}
    profile = _historical_profile(draws)
    last_draw = draws[-1]
    freq_recent = _recent_freq(draws, window=100)
    delays = _delays(draws)
    pair_freq = _pair_counter(draws)
    common_signatures = _common_range_signatures(draws)
    context_model = build_context_model(df, draw_hour=draw_hour, draw_minute=draw_minute)
    rng = random.Random(seed)

    scored: Dict[Tuple[int, ...], Dict[str, object]] = {}
    profile_cycle = [
        "soma_baixa",
        "soma_media",
        "soma_alta",
        "sequencia_forte",
        "visual_forte",
        "contrarian_controlado",
        "faixa_alta_reforcada",
        "assinatura_historica",
    ]

    for idx in range(max(1, int(candidate_pool))):
        if rng.random() < 0.40:
            profile_name = profile_cycle[idx % len(profile_cycle)]
            nums = _profiled_candidate(
                rng,
                profile_name=profile_name,
                profile=profile,
                common_signatures=common_signatures,
                freq_recent=freq_recent,
                delays=delays,
            )
        elif rng.random() < 0.70:
            profile_name = "weighted_temporal"
            nums = _weighted_candidate(rng, freq_recent, delays)
        else:
            profile_name = "monte_carlo_filtrado"
            nums = _random_candidate(rng)
        key = tuple(nums)
        if key in existing:
            continue
        row = score_candidate(
            nums,
            profile=profile,
            last_draw=last_draw,
            freq_recent=freq_recent,
            pair_freq=pair_freq,
            context_model=context_model,
            common_signatures=common_signatures,
        )
        row["metodo"] = profile_name
        scored[key] = row

    ranked_seed = sorted(scored.items(), key=lambda item: float(item[1]["score_final"]), reverse=True)
    parents = [list(key) for key, _ in ranked_seed[: max(2, int(population))]]

    for _generation in range(max(0, int(generations))):
        if len(parents) < 2:
            break
        children: List[List[int]] = []
        for _ in range(max(1, int(population))):
            parent_a, parent_b = rng.sample(parents, 2)
            children.append(_child_from_parents(rng, parent_a, parent_b))
        for nums in children:
            key = tuple(nums)
            if key in existing:
                continue
            row = score_candidate(
                nums,
                profile=profile,
                last_draw=last_draw,
                freq_recent=freq_recent,
                pair_freq=pair_freq,
                context_model=context_model,
                common_signatures=common_signatures,
            )
            row["metodo"] = "genetico_simples"
            if key not in scored or float(row["score_final"]) > float(scored[key]["score_final"]):
                scored[key] = row
        ranked = sorted(scored.items(), key=lambda item: float(item[1]["score_final"]), reverse=True)
        parents = [list(key) for key, _ in ranked[: max(2, int(population))]]

    candidates = pd.DataFrame(list(scored.values()))
    candidates = candidates.sort_values(["score_final", "nums"], ascending=[False, True]).head(max(1, int(top_games))).reset_index(drop=True)
    candidates.insert(0, "rank", range(1, len(candidates) + 1))

    summary = pd.DataFrame(
        [
            {"metrica": "candidate_pool_solicitado", "valor": int(candidate_pool)},
            {"metrica": "top_games", "valor": int(top_games)},
            {"metrica": "generations", "valor": int(generations)},
            {"metrica": "population", "valor": int(population)},
            {"metrica": "profiles_ativos", "valor": ",".join(profile_cycle)},
            {"metrica": "score_weights", "valor": "estatistico=0.27;historico=0.18;anti_popularidade=0.12;combinatorio=0.12;contextual=0.15;cenarios=0.10;contrarian=0.06"},
            {"metrica": "draw_hour_brasilia", "valor": int(draw_hour)},
            {"metrica": "draw_minute_brasilia", "valor": int(draw_minute)},
            {"metrica": "data_proximo_concurso", "valor": context_model.target.data_proximo_concurso},
            {"metrica": "dia_semana_proximo_concurso", "valor": context_model.target.dia_semana_nome},
            {"metrica": "fase_lua_proximo_concurso", "valor": context_model.target.fase_lua},
            {"metrica": "idade_lua_proximo_concurso", "valor": context_model.target.idade_lua},
            {"metrica": "iluminacao_lua_proximo_concurso", "valor": context_model.target.iluminacao_lua_percentual},
            {"metrica": "numerologia_data_raiz", "valor": context_model.target.numerologia_data_raiz},
            {"metrica": "numerologia_concurso_raiz", "valor": context_model.target.numerologia_concurso_raiz},
            {"metrica": "candidatos_unicos_avaliados", "valor": int(len(scored))},
            {"metrica": "melhor_score_final", "valor": float(candidates["score_final"].max()) if len(candidates) else 0.0},
            {"metrica": "ultimo_concurso_base", "valor": int(df["concurso"].max())},
        ]
    )
    return candidates, summary
