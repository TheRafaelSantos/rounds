from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .normalize import DEZENAS


MAX_DEZENA = 25
PICK_SIZE = 15


@dataclass(frozen=True)
class AuditoriaSummary:
    concursos: int
    resumo_rows: int
    dezenas_rows: int
    anomalias_rows: int
    monte_carlo_rows: int
    resumo_csv_path: str
    dezenas_csv_path: str
    anomalias_csv_path: str
    monte_carlo_csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Fase 6",
                "Acao: audit",
                f"Concursos auditados: {self.concursos}",
                f"Linhas resumo: {self.resumo_rows}",
                f"Linhas dezenas: {self.dezenas_rows}",
                f"Linhas anomalias: {self.anomalias_rows}",
                f"Linhas Monte Carlo: {self.monte_carlo_rows}",
                f"CSV resumo: {self.resumo_csv_path}",
                f"CSV dezenas: {self.dezenas_csv_path}",
                f"CSV anomalias: {self.anomalias_csv_path}",
                f"CSV Monte Carlo: {self.monte_carlo_csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Auditoria estatistica exploratoria gerada.",
            ]
        )


def _nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _chi_square_sf_approx(value: float, df: int) -> float:
    if value <= 0:
        return 1.0
    if df <= 0:
        return float("nan")
    z = ((value / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return max(0.0, min(1.0, _normal_sf(z)))


def _entropy_bits(counts: Sequence[int]) -> float:
    total = sum(int(c) for c in counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = float(count) / float(total)
        entropy -= p * math.log2(p)
    return entropy


def _zscore(value: float, mean: float, std: float) -> float:
    if std <= 1e-12 or not math.isfinite(std):
        return 0.0
    return (float(value) - float(mean)) / float(std)


def _overlaps(draws: Sequence[Sequence[int]]) -> List[int]:
    out: List[int] = []
    previous: set[int] | None = None
    for draw in draws:
        current = set(draw)
        if previous is not None:
            out.append(len(current & previous))
        previous = current
    return out


def _simulate_overlap_means(*, contests: int, runs: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(max(1, int(runs))):
        previous: set[int] | None = None
        overlaps: List[int] = []
        for _contest in range(contests):
            current = set(rng.sample(range(1, MAX_DEZENA + 1), PICK_SIZE))
            if previous is not None:
                overlaps.append(len(current & previous))
            previous = current
        means.append(float(sum(overlaps) / len(overlaps)) if overlaps else 0.0)
    return means


def build_auditoria(
    concursos: pd.DataFrame,
    *,
    monte_carlo_runs: int = 500,
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if concursos.empty:
        raise ValueError("Base de concursos vazia. Rode primeiro: python main.py --update")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    draws = [_nums_from_row(row) for _, row in df.iterrows()]
    contests = len(draws)
    total_drawn = contests * PICK_SIZE

    counts = {dezena: 0 for dezena in range(1, MAX_DEZENA + 1)}
    for draw in draws:
        for dezena in draw:
            counts[int(dezena)] += 1

    expected = total_drawn / MAX_DEZENA
    variance_per_dezena = contests * (PICK_SIZE / MAX_DEZENA) * (1.0 - PICK_SIZE / MAX_DEZENA)
    std_per_dezena = math.sqrt(variance_per_dezena)
    chi_square = sum(((counts[n] - expected) ** 2) / expected for n in range(1, MAX_DEZENA + 1))
    chi_p = _chi_square_sf_approx(chi_square, MAX_DEZENA - 1)
    entropy = _entropy_bits([counts[n] for n in range(1, MAX_DEZENA + 1)])
    max_entropy = math.log2(MAX_DEZENA)

    dezenas_rows: List[Dict[str, object]] = []
    for dezena in range(1, MAX_DEZENA + 1):
        observed = counts[dezena]
        dezenas_rows.append(
            {
                "dezena": dezena,
                "freq_observada": int(observed),
                "freq_esperada": round(float(expected), 6),
                "diferenca": round(float(observed - expected), 6),
                "z_score_aprox": round(_zscore(observed, expected, std_per_dezena), 6),
                "pct_total_sorteado": round(float(observed / total_drawn), 8),
            }
        )

    sums = [sum(draw) for draw in draws]
    pairs = [sum(1 for n in draw if n % 2 == 0) for draw in draws]
    overlaps = _overlaps(draws)
    sum_mean = float(pd.Series(sums).mean())
    sum_std = float(pd.Series(sums).std(ddof=0))
    pairs_mean = float(pd.Series(pairs).mean())
    pairs_std = float(pd.Series(pairs).std(ddof=0))
    overlap_mean = float(pd.Series(overlaps).mean()) if overlaps else 0.0
    overlap_std = float(pd.Series(overlaps).std(ddof=0)) if overlaps else 0.0

    anomalias_rows: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        concurso = int(row["concurso"])
        data_sorteio = str(row["data_sorteio"])
        z_sum = _zscore(sums[idx], sum_mean, sum_std)
        z_pairs = _zscore(pairs[idx], pairs_mean, pairs_std)
        if abs(z_sum) >= 2.75:
            anomalias_rows.append({"concurso": concurso, "data_sorteio": data_sorteio, "tipo": "soma_extrema", "valor": sums[idx], "z_score": round(z_sum, 6)})
        if abs(z_pairs) >= 2.75:
            anomalias_rows.append({"concurso": concurso, "data_sorteio": data_sorteio, "tipo": "paridade_extrema", "valor": pairs[idx], "z_score": round(z_pairs, 6)})
        if idx > 0:
            z_overlap = _zscore(overlaps[idx - 1], overlap_mean, overlap_std)
            if abs(z_overlap) >= 2.75:
                anomalias_rows.append(
                    {
                        "concurso": concurso,
                        "data_sorteio": data_sorteio,
                        "tipo": "repeticao_anterior_extrema",
                        "valor": overlaps[idx - 1],
                        "z_score": round(z_overlap, 6),
                    }
                )

    simulated_means = _simulate_overlap_means(contests=contests, runs=monte_carlo_runs, seed=seed)
    theoretical_overlap_mean = PICK_SIZE * PICK_SIZE / MAX_DEZENA
    real_distance = abs(overlap_mean - theoretical_overlap_mean)
    empirical_p = sum(1 for value in simulated_means if abs(value - theoretical_overlap_mean) >= real_distance) / len(simulated_means)
    monte_carlo = pd.DataFrame(
        [{"run": idx + 1, "media_repeticao_simulada": round(value, 8)} for idx, value in enumerate(simulated_means)]
    )

    resumo = pd.DataFrame(
        [
            {"metrica": "concursos", "valor": contests, "observacao": ""},
            {"metrica": "total_dezenas_sorteadas", "valor": total_drawn, "observacao": ""},
            {"metrica": "freq_esperada_por_dezena", "valor": round(float(expected), 6), "observacao": "15/25 por concurso"},
            {"metrica": "chi_square_dezenas", "valor": round(float(chi_square), 6), "observacao": "frequencia marginal das dezenas"},
            {"metrica": "chi_square_df", "valor": MAX_DEZENA - 1, "observacao": ""},
            {"metrica": "chi_square_p_value_aprox", "valor": round(float(chi_p), 8), "observacao": "aproximacao Wilson-Hilferty"},
            {"metrica": "entropia_dezenas_bits", "valor": round(float(entropy), 8), "observacao": ""},
            {"metrica": "entropia_maxima_bits", "valor": round(float(max_entropy), 8), "observacao": "log2(25)"},
            {"metrica": "entropia_ratio", "valor": round(float(entropy / max_entropy), 8), "observacao": ""},
            {"metrica": "media_soma", "valor": round(float(sum_mean), 6), "observacao": ""},
            {"metrica": "media_pares", "valor": round(float(pairs_mean), 6), "observacao": ""},
            {"metrica": "media_repeticao_anterior", "valor": round(float(overlap_mean), 6), "observacao": ""},
            {"metrica": "media_repeticao_teorica", "valor": round(float(theoretical_overlap_mean), 6), "observacao": "15*15/25"},
            {"metrica": "monte_carlo_runs", "valor": int(len(simulated_means)), "observacao": ""},
            {"metrica": "monte_carlo_p_value_empirico", "valor": round(float(empirical_p), 8), "observacao": "repeticao media vs sorteios simulados"},
        ]
    )

    return resumo, pd.DataFrame(dezenas_rows), pd.DataFrame(anomalias_rows), monte_carlo
