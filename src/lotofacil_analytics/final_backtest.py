from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .backtest_lotofacil import compute_hits, generate_random, nums_from_row
from .optimizer import build_optimized_candidates
from .predictor import select_final_games
from .post_result_analysis import format_nums, parse_numbers


@dataclass(frozen=True)
class FinalBacktestSummary:
    rows: int
    contests: int
    first_concurso: int
    last_concurso: int
    results_csv_path: str
    summary_csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Backtest Score Final",
                f"Concursos avaliados: {self.contests}",
                f"Primeiro concurso avaliado: {self.first_concurso}",
                f"Ultimo concurso avaliado: {self.last_concurso}",
                f"Linhas de resultado: {self.rows}",
                f"CSV resultados: {self.results_csv_path}",
                f"CSV resumo: {self.summary_csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Metodo final completo testado contra baseline aleatorio sem dados futuros.",
            ]
        )


def _score_result(
    *,
    modelo_nome: str,
    concurso: int,
    data_sorteio: str,
    real: Sequence[int],
    games: Sequence[Sequence[int]],
    generated_at: str,
    seed: int,
    extra: Dict[str, object],
) -> Dict[str, object]:
    hits = [compute_hits(game, real) for game in games]
    best_idx = max(range(len(hits)), key=lambda idx: hits[idx])
    union_hits = len(set().union(*(set(game) for game in games)) & set(real))
    return {
        "modelo_nome": modelo_nome,
        "versao_modelo": "ensemble_score_v2_backtest",
        "data_geracao_jogo": generated_at,
        "concurso_previsto": int(concurso),
        "data_sorteio": data_sorteio,
        "jogo_1": format_nums(games[0]),
        "jogo_2": format_nums(games[1]) if len(games) > 1 else "",
        "numeros_reais": format_nums(real),
        "acertos_jogo_1": int(hits[0]),
        "acertos_jogo_2": int(hits[1]) if len(hits) > 1 else None,
        "melhor_acerto_entre_2": int(hits[best_idx]),
        "melhor_jogo": int(best_idx + 1),
        "cobertura_uniao_2_jogos": int(union_hits),
        "acertou_11_em_algum": int(max(hits) >= 11),
        "acertou_12_em_algum": int(max(hits) >= 12),
        "acertou_13_em_algum": int(max(hits) >= 13),
        "acertou_14_em_algum": int(max(hits) >= 14),
        "acertou_15_em_algum": int(max(hits) >= 15),
        "seed_randomica": int(seed),
        **extra,
    }


def _random_two_games(rng: random.Random, existing: set[Tuple[int, ...]]) -> List[List[int]]:
    first = generate_random(rng, existing)
    existing_with_first = set(existing)
    existing_with_first.add(tuple(first))
    second = generate_random(rng, existing_with_first)
    return [first, second]


def run_final_score_backtest(
    concursos: pd.DataFrame,
    *,
    n_eval: int = 60,
    min_history: int = 300,
    seed: int = 123,
    candidate_pool: int = 2500,
    top_games: int = 60,
    generations: int = 6,
    population: int = 40,
    max_overlap: int = 8,
    draw_hour: int = 20,
    draw_minute: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if concursos.empty:
        raise ValueError("Base de concursos vazia. Rode primeiro: python main.py --update")
    if n_eval <= 0:
        raise ValueError("n_eval deve ser maior que zero.")
    if min_history < 10:
        raise ValueError("min_history deve ser pelo menos 10.")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    draws = [nums_from_row(row) for _, row in df.iterrows()]
    total = len(df)
    if total <= min_history:
        raise ValueError(f"Historico insuficiente: {total} concursos para min_history={min_history}.")

    start_idx = max(int(min_history), total - int(n_eval))
    generated_at = datetime.now().isoformat(timespec="seconds")
    rows: List[Dict[str, object]] = []

    for idx in range(start_idx, total):
        train_df = df.iloc[:idx].copy()
        real = draws[idx]
        concurso = int(df.loc[idx, "concurso"])
        data_sorteio = str(df.loc[idx, "data_sorteio"])
        existing = {tuple(draw) for draw in draws[:idx]}
        rng = random.Random(int(seed) * 1_000_003 + idx)

        candidates, _summary = build_optimized_candidates(
            train_df,
            seed=int(seed) * 10_007 + idx,
            candidate_pool=candidate_pool,
            top_games=max(top_games, 20),
            generations=generations,
            population=population,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        final_games_df = select_final_games(candidates, max_overlap=max_overlap)
        final_games = [parse_numbers(str(row["nums"])) for _, row in final_games_df.iterrows()]
        rows.append(
            _score_result(
                modelo_nome="ensemble_score_v2",
                concurso=concurso,
                data_sorteio=data_sorteio,
                real=real,
                games=final_games,
                generated_at=generated_at,
                seed=seed,
                extra={
                    "candidate_pool": int(candidate_pool),
                    "generations": int(generations),
                    "population": int(population),
                    "max_overlap": int(max_overlap),
                },
            )
        )

        random_games = _random_two_games(rng, existing)
        rows.append(
            _score_result(
                modelo_nome="baseline_2_jogos_aleatorios",
                concurso=concurso,
                data_sorteio=data_sorteio,
                real=real,
                games=random_games,
                generated_at=generated_at,
                seed=seed,
                extra={
                    "candidate_pool": None,
                    "generations": None,
                    "population": None,
                    "max_overlap": None,
                },
            )
        )

    results = pd.DataFrame(rows)
    summary_rows: List[Dict[str, object]] = []
    for model, group in results.groupby("modelo_nome"):
        best_hits = group["melhor_acerto_entre_2"].astype(int)
        game_1_hits = group["acertos_jogo_1"].astype(int)
        union_hits = group["cobertura_uniao_2_jogos"].astype(int)
        summary_rows.append(
            {
                "modelo_nome": model,
                "n_concursos": int(len(group)),
                "media_acertos_jogo_1": round(float(game_1_hits.mean()), 6),
                "media_melhor_acerto_entre_2": round(float(best_hits.mean()), 6),
                "media_cobertura_uniao_2_jogos": round(float(union_hits.mean()), 6),
                "max_melhor_acerto": int(best_hits.max()),
                "p_acertou_11_em_algum": round(float((best_hits >= 11).mean()), 6),
                "p_acertou_12_em_algum": round(float((best_hits >= 12).mean()), 6),
                "p_acertou_13_em_algum": round(float((best_hits >= 13).mean()), 6),
                "p_acertou_14_em_algum": round(float((best_hits >= 14).mean()), 6),
                "p_acertou_15_em_algum": round(float((best_hits >= 15).mean()), 6),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["media_melhor_acerto_entre_2", "modelo_nome"], ascending=[False, True]).reset_index(drop=True)
    return results, summary
