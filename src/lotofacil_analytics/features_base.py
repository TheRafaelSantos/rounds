from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pandas as pd

from .normalize import DEZENAS


PRIMOS = {2, 3, 5, 7, 11, 13, 17, 19, 23}
FIBONACCI = {1, 2, 3, 5, 8, 13, 21}
QUADRADOS_PERFEITOS = {1, 4, 9, 16, 25}


@dataclass(frozen=True)
class FeatureSummary:
    total_rows: int
    first_concurso: int
    last_concurso: int
    csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Fase 2",
                "Acao: features",
                f"Primeiro concurso: {self.first_concurso}",
                f"Ultimo concurso: {self.last_concurso}",
                f"Total de linhas: {self.total_rows}",
                f"CSV: {self.csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Features basicas geradas sem usar concursos futuros.",
            ]
        )


def _nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def _week_of_month(day: int) -> int:
    return int((day - 1) // 7 + 1)


def _count_range(nums: Sequence[int], start: int, end: int) -> int:
    return sum(1 for n in nums if start <= int(n) <= end)


def _line_counts(nums: Sequence[int]) -> Dict[str, int]:
    out = {f"linha_{i}": 0 for i in range(1, 6)}
    for n in nums:
        row = (int(n) - 1) // 5 + 1
        out[f"linha_{row}"] += 1
    return out


def _column_counts(nums: Sequence[int]) -> Dict[str, int]:
    out = {f"coluna_{i}": 0 for i in range(1, 6)}
    for n in nums:
        col = (int(n) - 1) % 5 + 1
        out[f"coluna_{col}"] += 1
    return out


def _max_consecutive_run(nums: Sequence[int]) -> int:
    ordered = sorted(int(n) for n in nums)
    best = cur = 1
    for idx in range(1, len(ordered)):
        if ordered[idx] == ordered[idx - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _sequence_count(nums: Sequence[int]) -> int:
    ordered = sorted(int(n) for n in nums)
    count = 0
    in_run = False
    for idx in range(1, len(ordered)):
        if ordered[idx] == ordered[idx - 1] + 1:
            if not in_run:
                count += 1
                in_run = True
        else:
            in_run = False
    return count


def _signature(prefix: str, values: Dict[str, int]) -> str:
    return " | ".join(f"{prefix}{key.split('_')[-1]}:{value}" for key, value in values.items())


def build_base_features(concursos: pd.DataFrame) -> pd.DataFrame:
    if concursos.empty:
        raise ValueError("Base de concursos vazia. Rode primeiro: python main.py --update")

    missing = [col for col in ["concurso", "data_sorteio", *DEZENAS] if col not in concursos.columns]
    if missing:
        raise ValueError(f"Base de concursos sem colunas obrigatorias: {missing}")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    df["data_sorteio"] = pd.to_datetime(df["data_sorteio"], errors="coerce")
    if df["data_sorteio"].isna().any():
        bad = df.loc[df["data_sorteio"].isna(), "concurso"].head(10).tolist()
        raise ValueError(f"Concursos com data_sorteio invalida: {bad}")

    rows: List[Dict[str, Any]] = []
    previous_nums: set[int] | None = None

    for _, row in df.iterrows():
        nums = _nums_from_row(row)
        nums_set = set(nums)
        date = row["data_sorteio"]
        gaps = [nums[idx] - nums[idx - 1] for idx in range(1, len(nums))]
        pares = sum(1 for n in nums if n % 2 == 0)
        impares = 15 - pares
        line_counts = _line_counts(nums)
        column_counts = _column_counts(nums)
        repeated = sorted(nums_set & previous_nums) if previous_nums is not None else []

        feature_row: Dict[str, Any] = {
            "concurso": int(row["concurso"]),
            "data_sorteio": date.date().isoformat(),
            "ano": int(date.year),
            "mes": int(date.month),
            "dia_mes": int(date.day),
            "dia_semana": int(date.dayofweek) + 1,
            "semana_mes": _week_of_month(int(date.day)),
            "quinzena": 1 if int(date.day) <= 15 else 2,
            "bimestre": int((date.month - 1) // 2 + 1),
            "trimestre": int(date.quarter),
            "semestre": int((date.month - 1) // 6 + 1),
            "qtd_pares": pares,
            "qtd_impares": impares,
            "soma_dezenas": int(sum(nums)),
            "media_dezenas": round(float(sum(nums) / len(nums)), 4),
            "menor_dezena": int(min(nums)),
            "maior_dezena": int(max(nums)),
            "amplitude": int(max(nums) - min(nums)),
            "qtd_primos": sum(1 for n in nums if n in PRIMOS),
            "qtd_fibonacci": sum(1 for n in nums if n in FIBONACCI),
            "qtd_quadrados_perfeitos": sum(1 for n in nums if n in QUADRADOS_PERFEITOS),
            "faixa_01_05": _count_range(nums, 1, 5),
            "faixa_06_10": _count_range(nums, 6, 10),
            "faixa_11_15": _count_range(nums, 11, 15),
            "faixa_16_20": _count_range(nums, 16, 20),
            "faixa_21_25": _count_range(nums, 21, 25),
            "qtd_01_12": _count_range(nums, 1, 12),
            "qtd_13_25": _count_range(nums, 13, 25),
            "gap_min": int(min(gaps)),
            "gap_max": int(max(gaps)),
            "gap_medio": round(float(sum(gaps) / len(gaps)), 4),
            "qtd_gaps_1": sum(1 for gap in gaps if gap == 1),
            "qtd_gaps_2": sum(1 for gap in gaps if gap == 2),
            "maior_sequencia_consecutiva": _max_consecutive_run(nums),
            "qtd_sequencias_consecutivas": _sequence_count(nums),
            "qtd_repetidas_anterior": len(repeated),
            "qtd_novas_vs_anterior": 15 - len(repeated) if previous_nums is not None else None,
            "dezenas_repetidas_anterior": " ".join(f"{n:02d}" for n in repeated),
            "assinatura_paridade": "-".join("P" if n % 2 == 0 else "I" for n in nums),
            "assinatura_faixas_5": (
                f"01_05:{_count_range(nums, 1, 5)} | "
                f"06_10:{_count_range(nums, 6, 10)} | "
                f"11_15:{_count_range(nums, 11, 15)} | "
                f"16_20:{_count_range(nums, 16, 20)} | "
                f"21_25:{_count_range(nums, 21, 25)}"
            ),
            "assinatura_linhas": _signature("L", line_counts),
            "assinatura_colunas": _signature("C", column_counts),
            "assinatura_gaps": "-".join(str(gap) for gap in gaps),
        }
        feature_row.update(line_counts)
        feature_row.update(column_counts)
        rows.append(feature_row)
        previous_nums = nums_set

    return pd.DataFrame(rows)
