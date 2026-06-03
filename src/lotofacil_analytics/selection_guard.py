from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

import pandas as pd


NUMBERS = tuple(range(1, 26))


def parse_nums(text: str) -> List[int]:
    nums = [int(part) for part in str(text).split()]
    if len(nums) != 15 or len(set(nums)) != 15 or any(n < 1 or n > 25 for n in nums):
        raise ValueError(f"Candidato invalido: {text}")
    return sorted(nums)


def format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def safe_float(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row:
        return default
    try:
        value = float(row.get(column, default))
    except (TypeError, ValueError):
        return default
    if pd.isna(value):
        return default
    return value


def _score_rank(rank: int) -> float:
    if rank <= 1:
        return 100.0
    return max(0.0, min(100.0, 100.0 / (1.0 + math.log10(max(1, int(rank))))))


def build_number_guard_table(candidates: pd.DataFrame, *, consensus_top: int = 1000) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    if "nums" not in candidates.columns or "score_final" not in candidates.columns:
        raise ValueError("Tabela de candidatos precisa ter colunas nums e score_final.")

    ranked = candidates.copy().sort_values(["score_final", "nums"], ascending=[False, True]).reset_index(drop=True)
    top_n = max(1, min(int(consensus_top), len(ranked)))
    top = ranked.head(top_n)
    all_count = max(1, len(ranked))
    top_counter: Counter[int] = Counter()
    all_counter: Counter[int] = Counter()
    parsed_all = [parse_nums(str(value)) for value in ranked["nums"].tolist()]
    for nums in parsed_all:
        all_counter.update(nums)
    for nums in parsed_all[:top_n]:
        top_counter.update(nums)

    top_score = float(pd.to_numeric(ranked["score_final"], errors="coerce").max())
    rows: List[Dict[str, object]] = []
    for n in NUMBERS:
        mask = ranked["nums"].map(lambda text: n in parse_nums(str(text)))
        number_rows = ranked.loc[mask].copy()
        if number_rows.empty:
            best_rank = len(ranked) + 1
            best_score = 0.0
        elif "rank" in number_rows.columns:
            best_rank = int(pd.to_numeric(number_rows["rank"], errors="coerce").min())
            best_score = float(pd.to_numeric(number_rows["score_final"], errors="coerce").max())
        else:
            best_rank = int(number_rows.index.min() + 1)
            best_score = float(pd.to_numeric(number_rows["score_final"], errors="coerce").max())
        freq_top = round(float(top_counter[n]) / top_n * 100.0, 6)
        freq_total = round(float(all_counter[n]) / all_count * 100.0, 6)
        rank_score = _score_rank(best_rank)
        score_gap = max(0.0, top_score - best_score)
        gap_score = max(0.0, min(100.0, 100.0 - score_gap * 60.0))
        underrepresented_bonus = max(0.0, 70.0 - freq_top)
        protection_score = round((0.34 * freq_total) + (0.30 * rank_score) + (0.24 * gap_score) + (0.12 * underrepresented_bonus), 6)
        if freq_top >= 70.0:
            category = "nucleo_consenso"
        elif best_rank <= 250 and freq_top <= 55.0:
            category = "risco_falso_negativo"
        elif best_rank <= 50 and freq_top <= 70.0:
            category = "risco_falso_negativo"
        else:
            category = "observacao"
        rows.append(
            {
                "dezena": n,
                "freq_top_consenso": freq_top,
                "freq_total_candidatos": freq_total,
                "melhor_rank": best_rank,
                "melhor_score_final": round(best_score, 6),
                "score_protecao_falso_negativo": protection_score,
                "categoria_guarda": category,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["categoria_guarda", "score_protecao_falso_negativo", "freq_top_consenso", "dezena"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def _guard_numbers(table: pd.DataFrame, category: str) -> List[int]:
    if table.empty:
        return []
    values = table.loc[table["categoria_guarda"] == category, "dezena"].tolist()
    return [int(value) for value in values]


def enrich_candidates_with_decision_guard(candidates: pd.DataFrame, *, consensus_top: int = 1000) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    guard_table = build_number_guard_table(candidates, consensus_top=consensus_top)
    consensus_scores = {
        int(row["dezena"]): float(row["freq_top_consenso"])
        for _, row in guard_table.iterrows()
    }
    risk_numbers = _guard_numbers(guard_table, "risco_falso_negativo")
    core_numbers = _guard_numbers(guard_table, "nucleo_consenso")
    risk_set = set(risk_numbers)
    core_set = set(core_numbers)
    risk_denominator = max(1, min(8, len(risk_numbers)))
    core_denominator = max(1, min(12, len(core_numbers)))

    rows: List[Dict[str, object]] = []
    for _, row in candidates.iterrows():
        out = row.to_dict()
        nums = parse_nums(str(row["nums"]))
        nums_set = set(nums)
        score_final = safe_float(row, "score_final")
        score_transicao = safe_float(row, "score_transicao", score_final)
        score_contextual = safe_float(row, "score_contextual", safe_float(row, "score_localidade_numerologia", score_final))
        score_climatico = safe_float(row, "score_climatico", 50.0)
        consensus_score = round(sum(consensus_scores.get(n, 0.0) for n in nums) / len(nums), 6)
        selected_risk = sorted(nums_set & risk_set)
        selected_core = sorted(nums_set & core_set)
        risk_coverage = round(min(100.0, len(selected_risk) / risk_denominator * 100.0), 6)
        core_coverage = round(min(100.0, len(selected_core) / core_denominator * 100.0), 6)
        protected_context = round(
            max(
                score_contextual,
                (score_contextual * 0.66) + (score_climatico * 0.04) + (consensus_score * 0.18) + (risk_coverage * 0.12),
            ),
            6,
        )
        decision_score = round(
            (score_final * 0.60)
            + (score_transicao * 0.12)
            + (protected_context * 0.10)
            + (consensus_score * 0.08)
            + (risk_coverage * 0.05)
            + (score_climatico * 0.05),
            6,
        )
        out["score_consenso_top"] = consensus_score
        out["score_contexto_protegido"] = protected_context
        out["score_cobertura_risco_falso_negativo"] = risk_coverage
        out["score_cobertura_nucleo_consenso"] = core_coverage
        out["qtd_dezenas_risco_falso_negativo"] = len(selected_risk)
        out["dezenas_risco_falso_negativo"] = format_nums(selected_risk)
        out["qtd_dezenas_nucleo_consenso"] = len(selected_core)
        out["dezenas_nucleo_consenso"] = format_nums(selected_core)
        out["score_decisao_protegida"] = decision_score
        out["criterio_contexto_protegido"] = "contexto_bonus_com_trava_anti_exclusao"
        rows.append(out)
    return pd.DataFrame(rows)


def select_guarded_best_candidate(candidates: pd.DataFrame, *, max_score_gap: float = 1.2) -> pd.Series:
    if candidates.empty:
        raise ValueError("Nenhum candidato disponivel para selecionar o jogo unico.")
    required = {"nums", "score_final"}
    if not required.issubset(candidates.columns):
        raise ValueError("Tabela de candidatos precisa ter colunas nums e score_final.")

    enriched = enrich_candidates_with_decision_guard(candidates)
    top_score = float(pd.to_numeric(enriched["score_final"], errors="coerce").max())
    eligible = enriched[pd.to_numeric(enriched["score_final"], errors="coerce") >= top_score - float(max_score_gap)].copy()
    if eligible.empty:
        eligible = enriched.copy()
    ranked = eligible.sort_values(
        ["score_decisao_protegida", "score_final", "nums"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best = ranked.iloc[0].copy()
    parse_nums(str(best["nums"]))
    best["criterio_selecao"] = f"decisao_protegida_gap<={float(max_score_gap):.2f}"
    return best
