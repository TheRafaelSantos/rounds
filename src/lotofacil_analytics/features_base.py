from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pandas as pd

from .normalize import DEZENAS


PRIMOS = {2, 3, 5, 7, 11, 13, 17, 19, 23}
FIBONACCI = {1, 2, 3, 5, 8, 13, 21}
QUADRADOS_PERFEITOS = {1, 4, 9, 16, 25}
LOTOFACIL_DEZENAS = list(range(1, 26))
NOMES_MESES = {
    1: "janeiro",
    2: "fevereiro",
    3: "marco",
    4: "abril",
    5: "maio",
    6: "junho",
    7: "julho",
    8: "agosto",
    9: "setembro",
    10: "outubro",
    11: "novembro",
    12: "dezembro",
}
NOMES_DIAS = {
    1: "segunda-feira",
    2: "terca-feira",
    3: "quarta-feira",
    4: "quinta-feira",
    5: "sexta-feira",
    6: "sabado",
    7: "domingo",
}
FERIADOS_FIXOS_NACIONAIS = {
    (1, 1),
    (4, 21),
    (5, 1),
    (9, 7),
    (10, 12),
    (11, 2),
    (11, 15),
    (12, 25),
}


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
                "Mensagem: Features basicas, temporais, visuais, repeticao, frequencia e atraso geradas sem usar concursos futuros.",
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


def _run_lengths(nums: Sequence[int]) -> List[int]:
    ordered = sorted(int(n) for n in nums)
    if not ordered:
        return []
    runs: List[int] = []
    cur = 1
    for idx in range(1, len(ordered)):
        if ordered[idx] == ordered[idx - 1] + 1:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return runs


def _max_consecutive_run(nums: Sequence[int]) -> int:
    runs = _run_lengths(nums)
    return max(runs) if runs else 1


def _sequence_count(nums: Sequence[int]) -> int:
    return sum(1 for length in _run_lengths(nums) if length >= 2)


def _dezenas_com_vizinho(nums: Sequence[int]) -> int:
    nums_set = set(int(n) for n in nums)
    return sum(1 for n in nums_set if (n - 1 in nums_set) or (n + 1 in nums_set))


def _signature(prefix: str, values: Dict[str, int]) -> str:
    return " | ".join(f"{prefix}{key.split('_')[-1]}:{value}" for key, value in values.items())


def _top_keys(values: Dict[int, int], count: int, *, reverse: bool) -> set[int]:
    ordered = sorted(values.items(), key=lambda item: ((-item[1] if reverse else item[1]), item[0]))
    return set(key for key, _ in ordered[:count])


def _freq_in_window(previous_draws: Sequence[set[int]], dezena: int, window: int) -> int:
    if not previous_draws:
        return 0
    return sum(1 for draw in previous_draws[-window:] if dezena in draw)


def _gap_series(values: Sequence[int]) -> pd.Series:
    return pd.Series([int(v) for v in values], dtype="float64")


def _mean_gap(indices: List[int]) -> float | None:
    if len(indices) < 2:
        return None
    gaps = [indices[idx] - indices[idx - 1] - 1 for idx in range(1, len(indices))]
    return round(float(_gap_series(gaps).mean()), 6)


def _estacao_do_ano(date: pd.Timestamp) -> str:
    month_day = (int(date.month), int(date.day))
    if month_day >= (12, 21) or month_day < (3, 20):
        return "verao"
    if month_day < (6, 21):
        return "outono"
    if month_day < (9, 23):
        return "inverno"
    return "primavera"


def _safe_days_between(start: Any, end: Any) -> int | None:
    if start in (None, "") or end in (None, ""):
        return None
    left = pd.to_datetime(start, errors="coerce")
    right = pd.to_datetime(end, errors="coerce")
    if pd.isna(left) or pd.isna(right):
        return None
    return int((right - left).days)


def _fixed_holiday_flags(date: pd.Timestamp) -> Dict[str, int]:
    cur = (int(date.month), int(date.day))
    previous = date - pd.Timedelta(days=1)
    next_day = date + pd.Timedelta(days=1)
    return {
        "eh_feriado_nacional": int(cur in FERIADOS_FIXOS_NACIONAIS),
        "eh_vespera_feriado": int((int(next_day.month), int(next_day.day)) in FERIADOS_FIXOS_NACIONAIS),
        "eh_pos_feriado": int((int(previous.month), int(previous.day)) in FERIADOS_FIXOS_NACIONAIS),
    }


def _diagonal_counts(nums: Sequence[int]) -> Dict[str, int]:
    nums_set = set(int(n) for n in nums)
    diagonal_principal = {1, 7, 13, 19, 25}
    diagonal_secundaria = {5, 9, 13, 17, 21}
    return {
        "qtd_diagonal_principal": len(nums_set & diagonal_principal),
        "qtd_diagonal_secundaria": len(nums_set & diagonal_secundaria),
    }


def _frequency_stats(
    *,
    selected_nums: Sequence[int],
    concursos_anteriores: int,
    total_counts: Dict[int, int],
    window_counts: Dict[int, Dict[int, int]],
) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "media_freq_total_das_15_dezenas": round(float(pd.Series([total_counts[n] for n in selected_nums]).mean()), 6),
    }
    for window in [5, 10, 20, 50, 100]:
        values = [window_counts[window][n] for n in selected_nums]
        stats[f"media_freq_ultimos_{window}_das_15_dezenas"] = round(float(pd.Series(values).mean()), 6)

    top_5 = _top_keys(total_counts, 5, reverse=True)
    top_10 = _top_keys(total_counts, 10, reverse=True)
    bottom_5 = _top_keys(total_counts, 5, reverse=False)
    bottom_10 = _top_keys(total_counts, 10, reverse=False)
    nums_set = set(int(n) for n in selected_nums)
    stats.update(
        {
            "qtd_dezenas_top_5_mais_frequentes": len(nums_set & top_5),
            "qtd_dezenas_top_10_mais_frequentes": len(nums_set & top_10),
            "qtd_dezenas_bottom_5_menos_frequentes": len(nums_set & bottom_5),
            "qtd_dezenas_bottom_10_menos_frequentes": len(nums_set & bottom_10),
            "percentual_freq_total": round(
                float(sum(total_counts[n] for n in selected_nums) / max(1, concursos_anteriores * len(selected_nums))),
                8,
            ),
        }
    )
    return stats


def _delay_stats(
    *,
    selected_nums: Sequence[int],
    concursos_anteriores: int,
    last_seen_idx: Dict[int, int | None],
    appearances: Dict[int, List[int]],
) -> Dict[str, object]:
    atrasos = [
        concursos_anteriores if last_seen_idx[n] is None else concursos_anteriores - int(last_seen_idx[n]) - 1
        for n in selected_nums
    ]
    mean_gaps = [_mean_gap(appearances[n]) for n in selected_nums]
    mean_gaps_numeric = [float(v) for v in mean_gaps if v is not None]
    return {
        "media_atraso_das_15_dezenas": round(float(pd.Series(atrasos).mean()), 6),
        "maior_atraso_entre_as_15": int(max(atrasos)),
        "menor_atraso_entre_as_15": int(min(atrasos)),
        "mediana_atraso_das_15": round(float(pd.Series(atrasos).median()), 6),
        "desvio_padrao_atraso_das_15": round(float(pd.Series(atrasos).std(ddof=0)), 6),
        "qtd_dezenas_com_atraso_maior_que_2": sum(1 for v in atrasos if v > 2),
        "qtd_dezenas_com_atraso_maior_que_5": sum(1 for v in atrasos if v > 5),
        "qtd_dezenas_com_atraso_maior_que_10": sum(1 for v in atrasos if v > 10),
        "qtd_dezenas_com_atraso_maior_que_20": sum(1 for v in atrasos if v > 20),
        "media_atraso_historico_das_15": round(float(pd.Series(mean_gaps_numeric).mean()), 6) if mean_gaps_numeric else None,
    }


def build_base_features(concursos: pd.DataFrame) -> pd.DataFrame:
    if concursos.empty:
        raise ValueError("Base de concursos vazia. Rode primeiro: python main.py --update")

    missing = [col for col in ["concurso", "data_sorteio", *DEZENAS] if col not in concursos.columns]
    if missing:
        raise ValueError(f"Base de concursos sem colunas obrigatorias: {missing}")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    df["data_sorteio"] = pd.to_datetime(df["data_sorteio"], errors="coerce")
    if "data_proximo_concurso" in df.columns:
        df["data_proximo_concurso"] = pd.to_datetime(df["data_proximo_concurso"], errors="coerce")
    if df["data_sorteio"].isna().any():
        bad = df.loc[df["data_sorteio"].isna(), "concurso"].head(10).tolist()
        raise ValueError(f"Concursos com data_sorteio invalida: {bad}")

    rows: List[Dict[str, Any]] = []
    previous_nums: set[int] | None = None
    previous_draws: List[set[int]] = []
    previous_date: pd.Timestamp | None = None
    total_counts = {dezena: 0 for dezena in LOTOFACIL_DEZENAS}
    last_seen_idx: Dict[int, int | None] = {dezena: None for dezena in LOTOFACIL_DEZENAS}
    appearances: Dict[int, List[int]] = {dezena: [] for dezena in LOTOFACIL_DEZENAS}

    for draw_idx, row in df.iterrows():
        nums = _nums_from_row(row)
        nums_set = set(nums)
        date = row["data_sorteio"]
        next_date = row.get("data_proximo_concurso")
        gaps = [nums[idx] - nums[idx - 1] for idx in range(1, len(nums))]
        gap_values = _gap_series(gaps)
        nums_values = _gap_series(nums)
        pares = sum(1 for n in nums if n % 2 == 0)
        impares = 15 - pares
        line_counts = _line_counts(nums)
        column_counts = _column_counts(nums)
        diagonal_counts = _diagonal_counts(nums)
        repeated = sorted(nums_set & previous_nums) if previous_nums is not None else []
        runs = _run_lengths(nums)
        concursos_anteriores = int(draw_idx)
        window_counts = {
            window: {dezena: _freq_in_window(previous_draws, dezena, window) for dezena in LOTOFACIL_DEZENAS}
            for window in [5, 10, 20, 50, 100]
        }
        repeated_windows = {
            window: len(nums_set & set().union(*previous_draws[-window:])) if previous_draws else 0
            for window in [2, 3, 5, 10]
        }

        feature_row: Dict[str, Any] = {
            "concurso": int(row["concurso"]),
            "data_sorteio": date.date().isoformat(),
            "ano": int(date.year),
            "mes": int(date.month),
            "nome_mes": NOMES_MESES[int(date.month)],
            "dia_mes": int(date.day),
            "dia_semana": int(date.dayofweek) + 1,
            "dia_semana_numero": int(date.dayofweek) + 1,
            "dia_semana_nome": NOMES_DIAS[int(date.dayofweek) + 1],
            "semana_do_ano": int(date.isocalendar().week),
            "semana_mes": _week_of_month(int(date.day)),
            "quinzena": 1 if int(date.day) <= 15 else 2,
            "quinzena_do_mes": 1 if int(date.day) <= 15 else 2,
            "bimestre": int((date.month - 1) // 2 + 1),
            "trimestre": int(date.quarter),
            "quadrimestre": int((date.month - 1) // 4 + 1),
            "semestre": int((date.month - 1) // 6 + 1),
            "estacao_do_ano": _estacao_do_ano(date),
            "dias_desde_inicio_do_ano": int(date.dayofyear - 1),
            "dias_para_fim_do_ano": int((pd.Timestamp(year=int(date.year), month=12, day=31) - date).days),
            "eh_fim_de_semana": int(int(date.dayofweek) >= 5),
            "eh_mes_dezembro": int(int(date.month) == 12),
            "eh_concurso_especial": int(bool(row.get("indicador_concurso_especial", 0))),
            "dias_desde_concurso_anterior": None if previous_date is None else int((date - previous_date).days),
            "dias_ate_proximo_concurso": _safe_days_between(date, next_date),
            "qtd_pares": pares,
            "qtd_impares": impares,
            "soma_dezenas": int(sum(nums)),
            "media_dezenas": round(float(nums_values.mean()), 6),
            "mediana_dezenas": round(float(nums_values.median()), 6),
            "desvio_padrao_dezenas": round(float(nums_values.std(ddof=0)), 6),
            "variancia_dezenas": round(float(nums_values.var(ddof=0)), 6),
            "menor_dezena": int(min(nums)),
            "maior_dezena": int(max(nums)),
            "amplitude": int(max(nums) - min(nums)),
            "amplitude_maior_menos_menor": int(max(nums) - min(nums)),
            "qtd_primos": sum(1 for n in nums if n in PRIMOS),
            "qtd_multiplos_3": sum(1 for n in nums if n % 3 == 0),
            "qtd_multiplos_5": sum(1 for n in nums if n % 5 == 0),
            "qtd_fibonacci": sum(1 for n in nums if n in FIBONACCI),
            "qtd_quadrados_perfeitos": sum(1 for n in nums if n in QUADRADOS_PERFEITOS),
            "qtd_baixas_01_12": _count_range(nums, 1, 12),
            "qtd_altas_13_25": _count_range(nums, 13, 25),
            "faixa_01_05": _count_range(nums, 1, 5),
            "faixa_06_10": _count_range(nums, 6, 10),
            "faixa_11_15": _count_range(nums, 11, 15),
            "faixa_16_20": _count_range(nums, 16, 20),
            "faixa_21_25": _count_range(nums, 21, 25),
            "qtd_01_05": _count_range(nums, 1, 5),
            "qtd_06_10": _count_range(nums, 6, 10),
            "qtd_11_15": _count_range(nums, 11, 15),
            "qtd_16_20": _count_range(nums, 16, 20),
            "qtd_21_25": _count_range(nums, 21, 25),
            "qtd_01_12": _count_range(nums, 1, 12),
            "qtd_13_25": _count_range(nums, 13, 25),
            "qtd_01_08": _count_range(nums, 1, 8),
            "qtd_09_16": _count_range(nums, 9, 16),
            "qtd_17_25": _count_range(nums, 17, 25),
            "qtd_01_06": _count_range(nums, 1, 6),
            "qtd_07_12": _count_range(nums, 7, 12),
            "qtd_13_18": _count_range(nums, 13, 18),
            "qtd_19_25": _count_range(nums, 19, 25),
            "linhas_ocupadas": sum(1 for value in line_counts.values() if value > 0),
            "colunas_ocupadas": sum(1 for value in column_counts.values() if value > 0),
            "max_dezenas_mesma_linha": max(line_counts.values()),
            "max_dezenas_mesma_coluna": max(column_counts.values()),
            "tem_linha_cheia": int(any(value == 5 for value in line_counts.values())),
            "tem_coluna_cheia": int(any(value == 5 for value in column_counts.values())),
            "tem_padrao_visual_forte": int(
                any(value == 5 for value in line_counts.values())
                or any(value == 5 for value in column_counts.values())
                or diagonal_counts["qtd_diagonal_principal"] >= 4
                or diagonal_counts["qtd_diagonal_secundaria"] >= 4
            ),
            "gap_min": int(min(gaps)),
            "gap_max": int(max(gaps)),
            "gap_medio": round(float(gap_values.mean()), 6),
            "gap_1_2": int(gaps[0]),
            "gap_2_3": int(gaps[1]),
            "gap_3_4": int(gaps[2]),
            "gap_4_5": int(gaps[3]),
            "gap_5_6": int(gaps[4]),
            "gap_6_7": int(gaps[5]),
            "gap_7_8": int(gaps[6]),
            "gap_8_9": int(gaps[7]),
            "gap_9_10": int(gaps[8]),
            "gap_10_11": int(gaps[9]),
            "gap_11_12": int(gaps[10]),
            "gap_12_13": int(gaps[11]),
            "gap_13_14": int(gaps[12]),
            "gap_14_15": int(gaps[13]),
            "menor_gap": int(min(gaps)),
            "maior_gap": int(max(gaps)),
            "media_gap": round(float(gap_values.mean()), 6),
            "mediana_gap": round(float(gap_values.median()), 6),
            "desvio_padrao_gap": round(float(gap_values.std(ddof=0)), 6),
            "qtd_gaps_1": sum(1 for gap in gaps if gap == 1),
            "qtd_gaps_2": sum(1 for gap in gaps if gap == 2),
            "qtd_gaps_ate_3": sum(1 for gap in gaps if gap <= 3),
            "tem_sequencia_2": int(any(length >= 2 for length in runs)),
            "tem_sequencia_3": int(any(length >= 3 for length in runs)),
            "tem_sequencia_4": int(any(length >= 4 for length in runs)),
            "tem_sequencia_5_ou_mais": int(any(length >= 5 for length in runs)),
            "maior_sequencia_consecutiva": _max_consecutive_run(nums),
            "qtd_dezenas_com_vizinho": _dezenas_com_vizinho(nums),
            "qtd_pares_consecutivos": sum(1 for gap in gaps if gap == 1),
            "qtd_blocos_consecutivos": _sequence_count(nums),
            "qtd_sequencias_consecutivas": _sequence_count(nums),
            "qtd_repetidas_concurso_anterior": len(repeated),
            "qtd_repetidas_anterior": len(repeated),
            "qtd_repetidas_ultimos_2_concursos": repeated_windows[2],
            "qtd_repetidas_ultimos_3_concursos": repeated_windows[3],
            "qtd_repetidas_ultimos_5_concursos": repeated_windows[5],
            "qtd_repetidas_ultimos_10_concursos": repeated_windows[10],
            "qtd_novas_vs_concurso_anterior": 15 - len(repeated) if previous_nums is not None else None,
            "qtd_novas_vs_anterior": 15 - len(repeated) if previous_nums is not None else None,
            "percentual_repeticao_concurso_anterior": round(float(len(repeated) / 15), 6) if previous_nums is not None else None,
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
        feature_row.update(_fixed_holiday_flags(date))
        feature_row.update(diagonal_counts)
        feature_row.update(_frequency_stats(selected_nums=nums, concursos_anteriores=concursos_anteriores, total_counts=total_counts, window_counts=window_counts))
        feature_row.update(_delay_stats(selected_nums=nums, concursos_anteriores=concursos_anteriores, last_seen_idx=last_seen_idx, appearances=appearances))
        for idx in range(1, 6):
            feature_row[f"qtd_linha_{idx}"] = line_counts[f"linha_{idx}"]
            feature_row[f"qtd_coluna_{idx}"] = column_counts[f"coluna_{idx}"]
        feature_row.update(line_counts)
        feature_row.update(column_counts)
        rows.append(feature_row)

        for dezena in nums_set:
            total_counts[dezena] += 1
            last_seen_idx[dezena] = draw_idx
            appearances[dezena].append(draw_idx)
        previous_nums = nums_set
        previous_draws.append(nums_set)
        previous_date = date

    return pd.DataFrame(rows)
