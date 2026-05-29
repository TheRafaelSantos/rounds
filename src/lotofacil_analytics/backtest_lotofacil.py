from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .normalize import DEZENAS


MAX_DEZENA = 25
PICK_SIZE = 15
MODEL_VERSION = "simple_walk_forward_v1"


@dataclass(frozen=True)
class BacktestSummary:
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
                "Resumo Lotofacil Analytics - Fase 5",
                "Acao: backtest",
                f"Concursos avaliados: {self.contests}",
                f"Primeiro concurso avaliado: {self.first_concurso}",
                f"Ultimo concurso avaliado: {self.last_concurso}",
                f"Linhas de resultado: {self.rows}",
                f"CSV resultados: {self.results_csv_path}",
                f"CSV resumo: {self.summary_csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Backtest walk-forward gerado sem usar concursos futuros.",
            ]
        )


def nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def compute_hits(predicted: Sequence[int], real: Sequence[int]) -> int:
    return len(set(int(n) for n in predicted) & set(int(n) for n in real))


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _counts(train_draws: Sequence[Sequence[int]], *, window: int | None = None) -> Counter[int]:
    draws = train_draws[-window:] if window else train_draws
    counter: Counter[int] = Counter()
    for draw in draws:
        counter.update(int(n) for n in draw)
    return counter


def _last_seen_delay(train_draws: Sequence[Sequence[int]]) -> Dict[int, int]:
    last_seen = {n: None for n in range(1, MAX_DEZENA + 1)}
    for idx, draw in enumerate(train_draws):
        for n in draw:
            last_seen[int(n)] = idx
    total = len(train_draws)
    return {n: total if last_seen[n] is None else total - int(last_seen[n]) - 1 for n in range(1, MAX_DEZENA + 1)}


def _top_by_count(counter: Counter[int], *, reverse: bool) -> List[int]:
    return [
        n
        for n, _ in sorted(
            ((n, int(counter.get(n, 0))) for n in range(1, MAX_DEZENA + 1)),
            key=lambda item: ((-item[1] if reverse else item[1]), item[0]),
        )
    ]


def generate_random(rng: random.Random, existing: set[Tuple[int, ...]]) -> List[int]:
    for _ in range(1000):
        nums = sorted(rng.sample(range(1, MAX_DEZENA + 1), PICK_SIZE))
        if tuple(nums) not in existing:
            return nums
    return sorted(rng.sample(range(1, MAX_DEZENA + 1), PICK_SIZE))


def generate_hot(train_draws: Sequence[Sequence[int]], *, window: int) -> List[int]:
    counter = _counts(train_draws, window=window)
    return sorted(_top_by_count(counter, reverse=True)[:PICK_SIZE])


def generate_cold(train_draws: Sequence[Sequence[int]]) -> List[int]:
    delay = _last_seen_delay(train_draws)
    ranked = sorted(delay.items(), key=lambda item: (-item[1], item[0]))
    return sorted(n for n, _ in ranked[:PICK_SIZE])


def generate_hybrid(train_draws: Sequence[Sequence[int]], *, window: int) -> List[int]:
    hot = generate_hot(train_draws, window=window)
    cold = generate_cold(train_draws)
    selected: List[int] = []
    selected.extend(hot[:7])
    selected.extend(n for n in cold if n not in selected and len(selected) < 14)
    total_rank = _top_by_count(_counts(train_draws), reverse=True)
    selected.extend(n for n in total_rank if n not in selected and len(selected) < PICK_SIZE)
    selected.extend(n for n in range(1, MAX_DEZENA + 1) if n not in selected and len(selected) < PICK_SIZE)
    return sorted(selected[:PICK_SIZE])


def _range_counts(nums: Sequence[int]) -> List[int]:
    return [sum(1 for n in nums if start <= n <= start + 4) for start in [1, 6, 11, 16, 21]]


def _max_run(nums: Sequence[int]) -> int:
    ordered = sorted(nums)
    best = cur = 1
    for idx in range(1, len(ordered)):
        if ordered[idx] == ordered[idx - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _balance_score(nums: Sequence[int], previous_draw: Sequence[int]) -> float:
    nums = sorted(int(n) for n in nums)
    pares = sum(1 for n in nums if n % 2 == 0)
    total = sum(nums)
    ranges = _range_counts(nums)
    overlap_previous = len(set(nums) & set(previous_draw))
    score = 0.0
    score += abs(pares - 7.5) * 2.0
    score += abs(total - 195.0) / 10.0
    score += sum(abs(value - 3) for value in ranges) * 1.5
    score += abs(overlap_previous - 9) * 1.0
    score += max(0, _max_run(nums) - 8) * 2.0
    return score


def generate_balanced(
    rng: random.Random,
    train_draws: Sequence[Sequence[int]],
    *,
    candidates: int,
    existing: set[Tuple[int, ...]],
) -> List[int]:
    previous_draw = train_draws[-1] if train_draws else []
    best_nums: List[int] | None = None
    best_score: float | None = None
    attempts = max(100, int(candidates))
    for _ in range(attempts):
        nums = sorted(rng.sample(range(1, MAX_DEZENA + 1), PICK_SIZE))
        if tuple(nums) in existing:
            continue
        score = _balance_score(nums, previous_draw)
        if best_score is None or score < best_score:
            best_nums = nums
            best_score = score
    return best_nums if best_nums is not None else generate_random(rng, existing)


def run_backtest(
    concursos: pd.DataFrame,
    *,
    n_eval: int = 300,
    min_history: int = 300,
    seed: int = 123,
    window: int = 100,
    candidates: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if concursos.empty:
        raise ValueError("Base de concursos vazia. Rode primeiro: python main.py --update")
    if n_eval <= 0:
        raise ValueError("n_eval deve ser maior que zero.")
    if min_history < 1:
        raise ValueError("min_history deve ser maior que zero.")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    draws = [nums_from_row(row) for _, row in df.iterrows()]
    total = len(df)
    if total <= min_history:
        raise ValueError(f"Historico insuficiente: {total} concursos para min_history={min_history}.")

    start_idx = max(int(min_history), total - int(n_eval))
    generated_at = datetime.now().isoformat(timespec="seconds")
    rows: List[Dict[str, object]] = []

    for idx in range(start_idx, total):
        train_draws = draws[:idx]
        real = draws[idx]
        concurso = int(df.loc[idx, "concurso"])
        data_sorteio = str(df.loc[idx, "data_sorteio"])
        existing = {tuple(draw) for draw in train_draws}
        rng = random.Random(int(seed) * 1_000_003 + idx)

        methods = {
            "aleatorio_puro": generate_random(rng, existing),
            "frequencia_quente": generate_hot(train_draws, window=window),
            "frequencia_fria": generate_cold(train_draws),
            "hibrido_quente_frio": generate_hybrid(train_draws, window=window),
            "balanceado_basico": generate_balanced(rng, train_draws, candidates=candidates, existing=existing),
        }

        for method, nums in methods.items():
            hits = compute_hits(nums, real)
            rows.append(
                {
                    "modelo_nome": method,
                    "versao_modelo": MODEL_VERSION,
                    "data_geracao_jogo": generated_at,
                    "concurso_previsto": concurso,
                    "data_sorteio": data_sorteio,
                    "numeros_sugeridos": _format_nums(nums),
                    "numeros_reais": _format_nums(real),
                    "qtd_acertos": int(hits),
                    "acertou_11": int(hits >= 11),
                    "acertou_12": int(hits >= 12),
                    "acertou_13": int(hits >= 13),
                    "acertou_14": int(hits >= 14),
                    "acertou_15": int(hits >= 15),
                    "metodo_geracao": method,
                    "seed_randomica": int(seed),
                    "janela_frequencia": int(window),
                    "candidatos_balanceado": int(candidates),
                    "observacao": "walk_forward_sem_dados_futuros",
                }
            )

    results = pd.DataFrame(rows)
    summary_rows: List[Dict[str, object]] = []
    for method, group in results.groupby("modelo_nome"):
        hits = group["qtd_acertos"].astype(int)
        summary_rows.append(
            {
                "modelo_nome": method,
                "n_jogos": int(len(group)),
                "media_acertos": round(float(hits.mean()), 6),
                "min_acertos": int(hits.min()),
                "max_acertos": int(hits.max()),
                "p_acertou_11": round(float((hits >= 11).mean()), 6),
                "p_acertou_12": round(float((hits >= 12).mean()), 6),
                "p_acertou_13": round(float((hits >= 13).mean()), 6),
                "p_acertou_14": round(float((hits >= 14).mean()), 6),
                "p_acertou_15": round(float((hits >= 15).mean()), 6),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["media_acertos", "modelo_nome"], ascending=[False, True]).reset_index(drop=True)
    return results, summary
