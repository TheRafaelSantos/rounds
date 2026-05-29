from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .normalize import DEZENAS


Combo = Tuple[int, ...]


@dataclass(frozen=True)
class CombinacoesSummary:
    concursos: int
    features_rows: int
    pares_rows: int
    trios_rows: int
    quartetos_rows: int
    first_concurso: int
    last_concurso: int
    features_csv_path: str
    pares_csv_path: str
    trios_csv_path: str
    quartetos_csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Fase 4",
                "Acao: combinacoes",
                f"Concursos: {self.concursos}",
                f"Primeiro concurso: {self.first_concurso}",
                f"Ultimo concurso: {self.last_concurso}",
                f"Linhas features: {self.features_rows}",
                f"Linhas pares: {self.pares_rows}",
                f"Linhas trios: {self.trios_rows}",
                f"Linhas quartetos: {self.quartetos_rows}",
                f"CSV features: {self.features_csv_path}",
                f"CSV pares: {self.pares_csv_path}",
                f"CSV trios: {self.trios_csv_path}",
                f"CSV quartetos: {self.quartetos_csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Combinacoes e assinaturas geradas.",
            ]
        )


def _nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def _combo_key(combo: Combo) -> str:
    return "-".join(f"{n:02d}" for n in combo)


def _all_combos(size: int) -> List[Combo]:
    return list(combinations(range(1, 26), size))


def _draw_combos(nums: Sequence[int], size: int) -> List[Combo]:
    return list(combinations(sorted(int(n) for n in nums), size))


def _range_count(nums: Sequence[int], start: int, end: int) -> int:
    return sum(1 for n in nums if start <= int(n) <= end)


def _line_counts(nums: Sequence[int]) -> Dict[str, int]:
    out = {f"L{i}": 0 for i in range(1, 6)}
    for n in nums:
        out[f"L{(int(n) - 1) // 5 + 1}"] += 1
    return out


def _column_counts(nums: Sequence[int]) -> Dict[str, int]:
    out = {f"C{i}": 0 for i in range(1, 6)}
    for n in nums:
        out[f"C{(int(n) - 1) % 5 + 1}"] += 1
    return out


def _signature_from_counts(values: Dict[str, int]) -> str:
    return " | ".join(f"{key}:{value}" for key, value in values.items())


def _gap_signature(nums: Sequence[int]) -> str:
    ordered = sorted(int(n) for n in nums)
    gaps = [ordered[idx] - ordered[idx - 1] for idx in range(1, len(ordered))]
    return "-".join(str(gap) for gap in gaps)


def _mod_signature(nums: Sequence[int], modulo: int) -> str:
    counts = {f"M{i}": 0 for i in range(modulo)}
    for n in nums:
        counts[f"M{int(n) % modulo}"] += 1
    return _signature_from_counts(counts)


def _combo_stats(combos: Iterable[Combo], counter: Counter[Combo]) -> Dict[str, float | int]:
    combo_list = list(combos)
    values = [int(counter.get(combo, 0)) for combo in combo_list]
    total = len(values)
    ineditos = sum(1 for value in values if value == 0)
    return {
        "qtd": total,
        "qtd_ineditos_ate_entao": ineditos,
        "pct_ineditos_ate_entao": round(float(ineditos / total), 6) if total else 0.0,
        "media_freq_ate_anterior": round(float(sum(values) / total), 6) if total else 0.0,
        "maior_freq_ate_anterior": max(values) if values else 0,
        "menor_freq_ate_anterior": min(values) if values else 0,
    }


def build_combinacoes_features(concursos: pd.DataFrame) -> pd.DataFrame:
    if concursos.empty:
        raise ValueError("Base de concursos vazia. Rode primeiro: python main.py --update")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    pair_counter: Counter[Combo] = Counter()
    trio_counter: Counter[Combo] = Counter()
    quarteto_counter: Counter[Combo] = Counter()
    previous_nums: set[int] | None = None
    rows: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        nums = _nums_from_row(row)
        repeated = sorted(set(nums) & previous_nums) if previous_nums is not None else []
        pair_combos = _draw_combos(nums, 2)
        trio_combos = _draw_combos(nums, 3)
        quarteto_combos = _draw_combos(nums, 4)
        pair_stats = _combo_stats(pair_combos, pair_counter)
        trio_stats = _combo_stats(trio_combos, trio_counter)
        quarteto_stats = _combo_stats(quarteto_combos, quarteto_counter)

        feature_row: Dict[str, object] = {
            "concurso": int(row["concurso"]),
            "data_sorteio": str(row["data_sorteio"]),
            "qtd_pares_combinatorios": int(pair_stats["qtd"]),
            "qtd_trios_combinatorios": int(trio_stats["qtd"]),
            "qtd_quartetos_combinatorios": int(quarteto_stats["qtd"]),
            "qtd_pares_ineditos_ate_entao": int(pair_stats["qtd_ineditos_ate_entao"]),
            "qtd_trios_ineditos_ate_entao": int(trio_stats["qtd_ineditos_ate_entao"]),
            "qtd_quartetos_ineditos_ate_entao": int(quarteto_stats["qtd_ineditos_ate_entao"]),
            "pct_pares_ineditos_ate_entao": pair_stats["pct_ineditos_ate_entao"],
            "pct_trios_ineditos_ate_entao": trio_stats["pct_ineditos_ate_entao"],
            "pct_quartetos_ineditos_ate_entao": quarteto_stats["pct_ineditos_ate_entao"],
            "media_freq_pares_ate_anterior": pair_stats["media_freq_ate_anterior"],
            "media_freq_trios_ate_anterior": trio_stats["media_freq_ate_anterior"],
            "media_freq_quartetos_ate_anterior": quarteto_stats["media_freq_ate_anterior"],
            "maior_freq_par_ate_anterior": int(pair_stats["maior_freq_ate_anterior"]),
            "maior_freq_trio_ate_anterior": int(trio_stats["maior_freq_ate_anterior"]),
            "maior_freq_quarteto_ate_anterior": int(quarteto_stats["maior_freq_ate_anterior"]),
            "assinatura_paridade": "-".join("P" if n % 2 == 0 else "I" for n in nums),
            "assinatura_faixas_5": (
                f"01_05:{_range_count(nums, 1, 5)} | "
                f"06_10:{_range_count(nums, 6, 10)} | "
                f"11_15:{_range_count(nums, 11, 15)} | "
                f"16_20:{_range_count(nums, 16, 20)} | "
                f"21_25:{_range_count(nums, 21, 25)}"
            ),
            "assinatura_linhas": _signature_from_counts(_line_counts(nums)),
            "assinatura_colunas": _signature_from_counts(_column_counts(nums)),
            "assinatura_gaps": _gap_signature(nums),
            "assinatura_mod_3": _mod_signature(nums, 3),
            "assinatura_mod_5": _mod_signature(nums, 5),
            "assinatura_repeticao_anterior": f"R:{len(repeated)} | dezenas:{' '.join(f'{n:02d}' for n in repeated)}",
        }
        rows.append(feature_row)

        pair_counter.update(pair_combos)
        trio_counter.update(trio_combos)
        quarteto_counter.update(quarteto_combos)
        previous_nums = set(nums)

    return pd.DataFrame(rows)


def _aggregate_combos(concursos: pd.DataFrame, size: int) -> pd.DataFrame:
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    total_counter: Counter[Combo] = Counter()
    recent_50_counter: Counter[Combo] = Counter()
    recent_100_counter: Counter[Combo] = Counter()
    first_seen: Dict[Combo, int] = {}
    last_seen: Dict[Combo, int] = {}
    last_concurso = int(df["concurso"].max())

    for _, row in df.iterrows():
        concurso = int(row["concurso"])
        combos = _draw_combos(_nums_from_row(row), size)
        total_counter.update(combos)
        for combo in combos:
            first_seen.setdefault(combo, concurso)
            last_seen[combo] = concurso

    for _, row in df.tail(50).iterrows():
        recent_50_counter.update(_draw_combos(_nums_from_row(row), size))
    for _, row in df.tail(100).iterrows():
        recent_100_counter.update(_draw_combos(_nums_from_row(row), size))

    rows: List[Dict[str, object]] = []
    for combo in _all_combos(size):
        record: Dict[str, object] = {
            "combo": _combo_key(combo),
            "tamanho": size,
            "freq_total_historico": int(total_counter.get(combo, 0)),
            "freq_ultimos_50": int(recent_50_counter.get(combo, 0)),
            "freq_ultimos_100": int(recent_100_counter.get(combo, 0)),
            "primeiro_concurso": first_seen.get(combo),
            "ultimo_concurso": last_seen.get(combo),
            "concursos_desde_ultima_saida": None if combo not in last_seen else last_concurso - int(last_seen[combo]),
        }
        for idx, dezena in enumerate(combo, start=1):
            record[f"d{idx}"] = int(dezena)
        rows.append(record)

    out = pd.DataFrame(rows)
    return out.sort_values(["freq_total_historico", "combo"], ascending=[False, True]).reset_index(drop=True)


def build_combinacoes_outputs(concursos: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        build_combinacoes_features(concursos),
        _aggregate_combos(concursos, 2),
        _aggregate_combos(concursos, 3),
        _aggregate_combos(concursos, 4),
    )
