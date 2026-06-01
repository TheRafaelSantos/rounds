from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .backtest_lotofacil import (
    PICK_SIZE,
    generate_balanced,
    generate_cold,
    generate_hot,
    generate_hybrid,
    generate_random,
    nums_from_row,
)
from .context_features import build_context_model
from .optimizer import _common_range_signatures, _historical_profile, build_optimized_candidates, score_candidate


SUPPORTED_GAME_METHODS = {
    "aleatorio_puro",
    "balanceado_basico",
    "frequencia_quente",
    "frequencia_fria",
    "hibrido_quente_frio",
    "score_equilibrado",
    "anti_popularidade_humana",
    "monte_carlo_filtrado",
    "genetico_opcional",
}


@dataclass(frozen=True)
class GeneratedGamesSummary:
    rows: int
    method: str
    csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Jogos Gerados",
                f"Metodo: {self.method}",
                f"Jogos gerados: {self.rows}",
                f"CSV: {self.csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Jogos gerados e validados dentro das regras da Lotofacil.",
            ]
        )


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _validate_nums(nums: Sequence[int]) -> List[int]:
    ordered = sorted(int(n) for n in nums)
    if len(ordered) != PICK_SIZE or len(set(ordered)) != PICK_SIZE or any(n < 1 or n > 25 for n in ordered):
        raise ValueError(f"Jogo invalido gerado: {ordered}")
    return ordered


def _historical_context(concursos: pd.DataFrame) -> Tuple[List[List[int]], set[Tuple[int, ...]], Dict[str, object]]:
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    draws = [nums_from_row(row) for _, row in df.iterrows()]
    existing = {tuple(draw) for draw in draws}
    return draws, existing, {"ultimo_concurso_base": int(df["concurso"].max())}


def _candidate_score_row(nums: Sequence[int], concursos: pd.DataFrame, *, draw_hour: int, draw_minute: int) -> Dict[str, object]:
    from collections import Counter
    from itertools import combinations

    draws = [nums_from_row(row) for _, row in concursos.copy().sort_values("concurso").iterrows()]
    profile = _historical_profile(draws)
    common_signatures = _common_range_signatures(draws)
    freq_recent: Counter[int] = Counter()
    for draw in draws[-100:]:
        freq_recent.update(draw)
    pair_freq: Counter[Tuple[int, int]] = Counter()
    for draw in draws:
        pair_freq.update(tuple(combo) for combo in combinations(sorted(draw), 2))
    context_model = build_context_model(concursos, draw_hour=draw_hour, draw_minute=draw_minute)
    return score_candidate(
        nums,
        profile=profile,
        last_draw=draws[-1],
        freq_recent=freq_recent,
        pair_freq=pair_freq,
        context_model=context_model,
        common_signatures=common_signatures,
    )


def generate_games(
    concursos: pd.DataFrame,
    *,
    method: str,
    qty: int,
    seed: int,
    window: int,
    candidates: int,
    candidate_pool: int,
    generations: int,
    population: int,
    draw_hour: int = 20,
    draw_minute: int = 0,
) -> pd.DataFrame:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
    if method not in SUPPORTED_GAME_METHODS:
        allowed = ", ".join(sorted(SUPPORTED_GAME_METHODS))
        raise ValueError(f"Metodo invalido: {method}. Permitidos: {allowed}")
    if qty <= 0:
        raise ValueError("qty deve ser maior que zero.")

    draws, existing, context = _historical_context(concursos)
    rng = random.Random(seed)
    generated_at = datetime.now().isoformat(timespec="seconds")
    rows: List[Dict[str, object]] = []
    seen = set(existing)

    if method in {"score_equilibrado", "anti_popularidade_humana", "monte_carlo_filtrado", "genetico_opcional"}:
        optimized, _summary = build_optimized_candidates(
            concursos,
            seed=seed,
            candidate_pool=max(candidate_pool, qty * 200),
            top_games=max(qty, 100),
            generations=generations,
            population=population,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        sort_column = "score_final"
        if method == "score_equilibrado":
            sort_column = "score_estatistico"
        elif method == "anti_popularidade_humana":
            sort_column = "score_anti_popularidade"
        ranked = optimized.sort_values([sort_column, "nums"], ascending=[False, True])
        if method == "monte_carlo_filtrado":
            ranked = ranked[ranked["metodo"] == "monte_carlo_filtrado"]
        elif method == "genetico_opcional":
            ranked = ranked[ranked["metodo"] == "genetico_simples"]
        for _, row in ranked.head(qty).iterrows():
            record = row.to_dict()
            record.update(
                {
                    "generated_at": generated_at,
                    "jogo": len(rows) + 1,
                    "metodo_geracao": method,
                    "seed_randomica": int(seed),
                    **context,
                }
            )
            rows.append(record)
        if len(rows) < qty:
            raise ValueError(f"Nao foi possivel gerar {qty} jogos unicos com o metodo {method}.")
        return pd.DataFrame(rows)

    attempts = 0
    while len(rows) < qty and attempts < max(5000, qty * 500):
        attempts += 1
        if method == "aleatorio_puro":
            nums = generate_random(rng, seen)
        elif method == "balanceado_basico":
            nums = generate_balanced(rng, draws, candidates=candidates, existing=seen)
        elif method == "frequencia_quente":
            nums = generate_hot(draws, window=window)
            if tuple(nums) in seen:
                nums = generate_balanced(rng, draws, candidates=candidates, existing=seen)
        elif method == "frequencia_fria":
            nums = generate_cold(draws)
            if tuple(nums) in seen:
                nums = generate_balanced(rng, draws, candidates=candidates, existing=seen)
        else:
            nums = generate_hybrid(draws, window=window)
            if tuple(nums) in seen:
                nums = generate_balanced(rng, draws, candidates=candidates, existing=seen)
        nums = _validate_nums(nums)
        key = tuple(nums)
        if key in seen:
            continue
        seen.add(key)
        score_row = _candidate_score_row(nums, concursos, draw_hour=draw_hour, draw_minute=draw_minute)
        score_row.update(
            {
                "generated_at": generated_at,
                "jogo": len(rows) + 1,
                "nums": _format_nums(nums),
                "metodo_geracao": method,
                "seed_randomica": int(seed),
                **context,
            }
        )
        rows.append(score_row)

    if len(rows) < qty:
        raise ValueError(f"Nao foi possivel gerar {qty} jogos unicos com o metodo {method}.")
    return pd.DataFrame(rows)
