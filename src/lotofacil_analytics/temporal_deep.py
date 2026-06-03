from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .normalize import DEZENAS


NUMBERS = tuple(range(1, 26))


@dataclass(frozen=True)
class TemporalDeepSummary:
    rows: int
    contests_processed: int
    contests_total: int
    csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Temporal Profundo",
                f"Linhas geradas: {self.rows}",
                f"Concursos processados nesta execucao: {self.contests_processed}",
                f"Concursos totais na base: {self.contests_total}",
                f"CSV: {self.csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Analises temporais profundas salvas de forma incremental.",
            ]
        )


def nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def bimestre_from_month(month: int) -> int:
    return int((int(month) - 1) // 2 + 1)


def _score_from_counter(counter: Counter[int], sample_size: int, *, shrink_base: float = 40.0) -> Dict[int, float]:
    if sample_size <= 0:
        return {n: 50.0 for n in NUMBERS}
    expected = sample_size * 15.0 / 25.0
    shrink = min(1.0, sample_size / shrink_base)
    scores: Dict[int, float] = {}
    for n in NUMBERS:
        raw = 50.0 + (float(counter.get(n, 0)) - expected) * 7.0
        scores[n] = round(max(0.0, min(100.0, 50.0 + (raw - 50.0) * shrink)), 6)
    return scores


def _counter_recent(previous_draws: Sequence[Tuple[pd.Timestamp, Sequence[int]]], target_date: pd.Timestamp, days: int) -> Counter[int]:
    start = target_date - pd.Timedelta(days=int(days))
    counter: Counter[int] = Counter()
    for draw_date, nums in previous_draws:
        if start <= draw_date < target_date:
            counter.update(int(n) for n in nums)
    return counter


def temporal_deep_number_scores(
    concursos: pd.DataFrame,
    *,
    target_date: pd.Timestamp | str,
) -> Dict[int, float]:
    if concursos.empty:
        return {n: 50.0 for n in NUMBERS}
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target):
        return {n: 50.0 for n in NUMBERS}
    df = concursos.copy()
    df["data_sorteio"] = pd.to_datetime(df["data_sorteio"], errors="coerce")
    df = df.dropna(subset=["data_sorteio"]).sort_values("concurso").reset_index(drop=True)
    df = df[df["data_sorteio"] < target].copy()
    if df.empty:
        return {n: 50.0 for n in NUMBERS}

    target_weekday = int(target.isoweekday())
    target_bimester = bimestre_from_month(int(target.month))
    target_quarter = int((target.month - 1) // 3 + 1)
    target_semester = int((target.month - 1) // 6 + 1)

    counters = {
        "weekday": Counter(),
        "bimestre": Counter(),
        "trimestre": Counter(),
        "semestre": Counter(),
        "recent_15d": Counter(),
        "recent_30d": Counter(),
    }
    samples = {key: 0 for key in counters}
    previous_draws: List[Tuple[pd.Timestamp, Sequence[int]]] = []

    for _, row in df.iterrows():
        draw_date = pd.Timestamp(row["data_sorteio"])
        nums = nums_from_row(row)
        previous_draws.append((draw_date, nums))
        if int(draw_date.isoweekday()) == target_weekday:
            counters["weekday"].update(nums)
            samples["weekday"] += 1
        if bimestre_from_month(int(draw_date.month)) == target_bimester:
            counters["bimestre"].update(nums)
            samples["bimestre"] += 1
        if int((draw_date.month - 1) // 3 + 1) == target_quarter:
            counters["trimestre"].update(nums)
            samples["trimestre"] += 1
        if int((draw_date.month - 1) // 6 + 1) == target_semester:
            counters["semestre"].update(nums)
            samples["semestre"] += 1

    for days, key in [(15, "recent_15d"), (30, "recent_30d")]:
        counters[key] = _counter_recent(previous_draws, target, days)
        samples[key] = max(0, sum(1 for draw_date, _nums in previous_draws if target - pd.Timedelta(days=days) <= draw_date < target))

    score_maps = {key: _score_from_counter(counter, samples[key]) for key, counter in counters.items()}
    weights = {
        "weekday": 0.22,
        "recent_15d": 0.18,
        "recent_30d": 0.16,
        "bimestre": 0.18,
        "trimestre": 0.13,
        "semestre": 0.13,
    }
    out: Dict[int, float] = {}
    for n in NUMBERS:
        out[n] = round(sum(weights[key] * score_maps[key][n] for key in weights), 6)
    return out


def selected_temporal_deep_score(nums: Iterable[int], scores: Dict[int, float]) -> float:
    selected = [int(n) for n in nums]
    if not selected:
        return 50.0
    return round(sum(float(scores.get(n, 50.0)) for n in selected) / len(selected), 6)


def build_temporal_deep_rows(concursos: pd.DataFrame, *, target_concursos: set[int] | None = None) -> pd.DataFrame:
    if concursos.empty:
        return pd.DataFrame()
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    df["data_sorteio"] = pd.to_datetime(df["data_sorteio"], errors="coerce")
    target_filter = set(int(value) for value in target_concursos) if target_concursos else None
    rows: List[Dict[str, object]] = []
    previous_draws: deque[Tuple[pd.Timestamp, Sequence[int]]] = deque()
    counters_weekday: Dict[int, Counter[int]] = defaultdict(Counter)
    counters_bimester: Dict[int, Counter[int]] = defaultdict(Counter)
    counters_quarter: Dict[int, Counter[int]] = defaultdict(Counter)
    counters_semester: Dict[int, Counter[int]] = defaultdict(Counter)
    samples_weekday: Counter[int] = Counter()
    samples_bimester: Counter[int] = Counter()
    samples_quarter: Counter[int] = Counter()
    samples_semester: Counter[int] = Counter()

    for _, row in df.iterrows():
        if pd.isna(row["data_sorteio"]):
            continue
        concurso = int(row["concurso"])
        draw_date = pd.Timestamp(row["data_sorteio"])
        weekday = int(draw_date.isoweekday())
        bimester = bimestre_from_month(int(draw_date.month))
        quarter = int((draw_date.month - 1) // 3 + 1)
        semester = int((draw_date.month - 1) // 6 + 1)
        recent_15 = _counter_recent(list(previous_draws), draw_date, 15)
        recent_30 = _counter_recent(list(previous_draws), draw_date, 30)
        samples_15 = sum(1 for prev_date, _nums in previous_draws if draw_date - pd.Timedelta(days=15) <= prev_date < draw_date)
        samples_30 = sum(1 for prev_date, _nums in previous_draws if draw_date - pd.Timedelta(days=30) <= prev_date < draw_date)

        if target_filter is None or concurso in target_filter:
            for dezena in NUMBERS:
                rows.append(
                    {
                        "concurso": concurso,
                        "data_sorteio": draw_date.date().isoformat(),
                        "dezena": dezena,
                        "dia_semana_numero": weekday,
                        "bimestre": bimester,
                        "trimestre": quarter,
                        "semestre": semester,
                        "freq_mesmo_dia_semana_ate_anterior": int(counters_weekday[weekday].get(dezena, 0)),
                        "amostra_mesmo_dia_semana_ate_anterior": int(samples_weekday[weekday]),
                        "freq_ultimos_15_dias_ate_anterior": int(recent_15.get(dezena, 0)),
                        "amostra_ultimos_15_dias_ate_anterior": int(samples_15),
                        "freq_ultimos_30_dias_ate_anterior": int(recent_30.get(dezena, 0)),
                        "amostra_ultimos_30_dias_ate_anterior": int(samples_30),
                        "freq_mesmo_bimestre_ate_anterior": int(counters_bimester[bimester].get(dezena, 0)),
                        "amostra_mesmo_bimestre_ate_anterior": int(samples_bimester[bimester]),
                        "freq_mesmo_trimestre_ate_anterior": int(counters_quarter[quarter].get(dezena, 0)),
                        "amostra_mesmo_trimestre_ate_anterior": int(samples_quarter[quarter]),
                        "freq_mesmo_semestre_ate_anterior": int(counters_semester[semester].get(dezena, 0)),
                        "amostra_mesmo_semestre_ate_anterior": int(samples_semester[semester]),
                        "saiu_no_concurso": int(dezena in set(nums_from_row(row))),
                    }
                )

        nums = nums_from_row(row)
        counters_weekday[weekday].update(nums)
        counters_bimester[bimester].update(nums)
        counters_quarter[quarter].update(nums)
        counters_semester[semester].update(nums)
        samples_weekday[weekday] += 1
        samples_bimester[bimester] += 1
        samples_quarter[quarter] += 1
        samples_semester[semester] += 1
        previous_draws.append((draw_date, nums))

    return pd.DataFrame(rows)


def summarize_temporal_deep(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame([{"metrica": "linhas", "valor": 0}])
    return pd.DataFrame(
        [
            {"metrica": "linhas", "valor": int(len(rows))},
            {"metrica": "concursos", "valor": int(rows["concurso"].nunique())},
            {"metrica": "primeiro_concurso", "valor": int(rows["concurso"].min())},
            {"metrica": "ultimo_concurso", "valor": int(rows["concurso"].max())},
            {"metrica": "dezenas", "valor": int(rows["dezena"].nunique())},
        ]
    )
