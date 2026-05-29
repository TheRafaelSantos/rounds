from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .features_base import FIBONACCI, PRIMOS, QUADRADOS_PERFEITOS
from .normalize import DEZENAS, ORDEM_SORTEIO


LOTOFACIL_DEZENAS = list(range(1, 26))


@dataclass(frozen=True)
class DezenasSummary:
    concursos: int
    long_rows: int
    historico_rows: int
    first_concurso: int
    last_concurso: int
    long_csv_path: str
    historico_csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Fase 3",
                "Acao: dezenas",
                f"Concursos: {self.concursos}",
                f"Primeiro concurso: {self.first_concurso}",
                f"Ultimo concurso: {self.last_concurso}",
                f"Linhas dezenas_long: {self.long_rows}",
                f"Linhas dezenas_historico: {self.historico_rows}",
                f"CSV dezenas_long: {self.long_csv_path}",
                f"CSV dezenas_historico: {self.historico_csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Historico por dezena gerado sem usar concursos futuros.",
            ]
        )


def _nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def _draw_order_from_row(row: pd.Series) -> List[int]:
    if all(col in row.index for col in ORDEM_SORTEIO):
        values = [row[col] for col in ORDEM_SORTEIO]
        if not any(pd.isna(v) for v in values):
            return [int(v) for v in values]
    return _nums_from_row(row)


def _static_dezena_features(dezena: int) -> Dict[str, int]:
    return {
        "dezena_par": int(dezena % 2 == 0),
        "dezena_prima": int(dezena in PRIMOS),
        "dezena_fibonacci": int(dezena in FIBONACCI),
        "dezena_quadrado_perfeito": int(dezena in QUADRADOS_PERFEITOS),
        "grupo_dezena_5": int((dezena - 1) // 5 + 1),
        "linha_volante": int((dezena - 1) // 5 + 1),
        "coluna_volante": int((dezena - 1) % 5 + 1),
    }


def _rank_desc(values: Dict[int, int]) -> Dict[int, int]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    return {dezena: rank for rank, (dezena, _) in enumerate(ordered, start=1)}


def _freq_in_window(previous_draws: Sequence[set[int]], dezena: int, window: int) -> int:
    if not previous_draws:
        return 0
    return sum(1 for draw in previous_draws[-window:] if dezena in draw)


def _mean_gap(indices: List[int]) -> Optional[float]:
    if len(indices) < 2:
        return None
    gaps = [indices[idx] - indices[idx - 1] - 1 for idx in range(1, len(indices))]
    return round(float(sum(gaps) / len(gaps)), 4)


def build_dezenas_long(concursos: pd.DataFrame) -> pd.DataFrame:
    if concursos.empty:
        raise ValueError("Base de concursos vazia. Rode primeiro: python main.py --update")

    rows: List[Dict[str, object]] = []
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    for _, row in df.iterrows():
        ordered = _nums_from_row(row)
        draw_order = _draw_order_from_row(row)
        ordered_pos = {dezena: idx for idx, dezena in enumerate(ordered, start=1)}
        draw_pos = {dezena: idx for idx, dezena in enumerate(draw_order, start=1)}
        for dezena in ordered:
            rows.append(
                {
                    "concurso": int(row["concurso"]),
                    "data_sorteio": str(row["data_sorteio"]),
                    "dezena": int(dezena),
                    "posicao_ordenada": int(ordered_pos[dezena]),
                    "posicao_sorteio": int(draw_pos.get(dezena, ordered_pos[dezena])),
                }
            )
    return pd.DataFrame(rows)


def build_dezenas_historico(concursos: pd.DataFrame) -> pd.DataFrame:
    if concursos.empty:
        raise ValueError("Base de concursos vazia. Rode primeiro: python main.py --update")

    missing = [col for col in ["concurso", "data_sorteio", *DEZENAS] if col not in concursos.columns]
    if missing:
        raise ValueError(f"Base de concursos sem colunas obrigatorias: {missing}")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    previous_draws: List[set[int]] = []
    total_counts = {dezena: 0 for dezena in LOTOFACIL_DEZENAS}
    last_seen_idx: Dict[int, Optional[int]] = {dezena: None for dezena in LOTOFACIL_DEZENAS}
    appearances: Dict[int, List[int]] = {dezena: [] for dezena in LOTOFACIL_DEZENAS}
    rows: List[Dict[str, object]] = []

    for draw_idx, row in df.iterrows():
        current_nums = set(_nums_from_row(row))
        concursos_anteriores = int(draw_idx)

        freq_5 = {dezena: _freq_in_window(previous_draws, dezena, 5) for dezena in LOTOFACIL_DEZENAS}
        freq_10 = {dezena: _freq_in_window(previous_draws, dezena, 10) for dezena in LOTOFACIL_DEZENAS}
        freq_20 = {dezena: _freq_in_window(previous_draws, dezena, 20) for dezena in LOTOFACIL_DEZENAS}
        freq_50 = {dezena: _freq_in_window(previous_draws, dezena, 50) for dezena in LOTOFACIL_DEZENAS}
        freq_100 = {dezena: _freq_in_window(previous_draws, dezena, 100) for dezena in LOTOFACIL_DEZENAS}
        atraso = {
            dezena: concursos_anteriores if last_seen_idx[dezena] is None else concursos_anteriores - int(last_seen_idx[dezena]) - 1
            for dezena in LOTOFACIL_DEZENAS
        }
        rank_freq_total = _rank_desc(total_counts)
        rank_freq_50 = _rank_desc(freq_50)
        rank_atraso = _rank_desc(atraso)

        previous_draw = previous_draws[-1] if previous_draws else set()
        last_5_union = set().union(*previous_draws[-5:]) if previous_draws else set()
        last_10_union = set().union(*previous_draws[-10:]) if previous_draws else set()

        for dezena in LOTOFACIL_DEZENAS:
            record: Dict[str, object] = {
                "concurso": int(row["concurso"]),
                "data_sorteio": str(row["data_sorteio"]),
                "dezena": int(dezena),
                "saiu_no_concurso": int(dezena in current_nums),
                "concursos_anteriores": concursos_anteriores,
                "freq_total_ate_anterior": int(total_counts[dezena]),
                "freq_ultimos_5": int(freq_5[dezena]),
                "freq_ultimos_10": int(freq_10[dezena]),
                "freq_ultimos_20": int(freq_20[dezena]),
                "freq_ultimos_50": int(freq_50[dezena]),
                "freq_ultimos_100": int(freq_100[dezena]),
                "saiu_concurso_anterior": int(dezena in previous_draw),
                "saiu_ultimos_5": int(dezena in last_5_union),
                "saiu_ultimos_10": int(dezena in last_10_union),
                "atraso_atual": int(atraso[dezena]),
                "nunca_saiu_ate_anterior": int(last_seen_idx[dezena] is None),
                "media_atraso_ate_anterior": _mean_gap(appearances[dezena]),
                "rank_freq_total": int(rank_freq_total[dezena]),
                "rank_freq_50": int(rank_freq_50[dezena]),
                "rank_atraso": int(rank_atraso[dezena]),
            }
            record.update(_static_dezena_features(dezena))
            rows.append(record)

        for dezena in current_nums:
            total_counts[dezena] += 1
            last_seen_idx[dezena] = draw_idx
            appearances[dezena].append(draw_idx)
        previous_draws.append(current_nums)

    return pd.DataFrame(rows)


def build_dezenas_outputs(concursos: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return build_dezenas_long(concursos), build_dezenas_historico(concursos)
