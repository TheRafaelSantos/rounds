# temporal_deep.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from data import DEZENAS
from features import (
    count_odds, count_primes, count_birthdays, sum_nums, amplitude, has_consecutive
)

NUMBERS = list(range(1, 61))


# -----------------------------
# Buckets temporais
# -----------------------------
def week_of_month(dt: pd.Timestamp) -> int:
    return int((dt.day - 1) // 7 + 1)  # 1..5

def day_phase(day: int) -> str:
    if day <= 10: return "inicio(1-10)"
    if day <= 20: return "meio(11-20)"
    return "fim(21-31)"

def day_window(day: int, center: int, radius: int) -> str:
    # exemplo: center=15, radius=2 => "win15"
    return f"win{center}" if abs(day - center) <= radius else "out"

def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df["data_sorteio"], errors="coerce")

    df["ano"] = d.dt.year.astype("Int64")
    df["mes"] = d.dt.month.astype("Int64")
    df["dia_mes"] = d.dt.day.astype("Int64")
    df["semana_mes"] = d.apply(lambda x: week_of_month(x) if pd.notna(x) else pd.NA).astype("Int64")

    df["quinzena"] = np.where(df["dia_mes"] <= 15, 1, 2).astype("Int64")
    df["bimestre"] = ((df["mes"] - 1) // 2 + 1).astype("Int64")
    df["trimestre"] = d.dt.quarter.astype("Int64")
    df["semestre"] = ((df["mes"] - 1) // 6 + 1).astype("Int64")

    df["bi_anual"] = ((df["ano"] // 2) * 2).astype("Int64")
    df["meia_decada"] = ((df["ano"] // 5) * 5).astype("Int64")
    df["decada"] = ((df["ano"] // 10) * 10).astype("Int64")

    df["fase_mes"] = df["dia_mes"].apply(lambda x: day_phase(int(x)) if pd.notna(x) else None).astype("string")
    df["dia15_win2"] = df["dia_mes"].apply(lambda x: day_window(int(x), 15, 2) if pd.notna(x) else None).astype("string")

    # “numerologia” opcional: raiz digital da soma (1..9, 0 fica 9)
    def digital_root(n: int) -> int:
        if n <= 0: return 0
        r = n % 9
        return 9 if r == 0 else r

    df["soma"] = df["nums"].apply(sum_nums).astype("Int64")
    df["soma_dr"] = df["soma"].apply(lambda x: digital_root(int(x)) if pd.notna(x) else pd.NA).astype("Int64")
    df["soma_mod9"] = df["soma"].apply(lambda x: int(x) % 9 if pd.notna(x) else pd.NA).astype("Int64")

    return df


# -----------------------------
# Contagens e métricas de coorte
# -----------------------------
def _counts_for_mask(df: pd.DataFrame, mask: np.ndarray) -> Tuple[np.ndarray, int]:
    sub = df.loc[mask, DEZENAS]
    n = int(len(sub))
    counts = np.zeros(60, dtype=float)
    if n == 0:
        return counts, 0
    arr = sub.to_numpy(dtype=int)
    for row in arr:
        for x in row:
            if 1 <= int(x) <= 60:
                counts[int(x) - 1] += 1
    return counts, n

def _rates(counts: np.ndarray, n_draws: int) -> np.ndarray:
    if n_draws <= 0:
        return np.ones(60) / 60.0
    # cada sorteio tem 6 dezenas, então rate por sorteio:
    return counts / float(n_draws)

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # Jensen-Shannon divergence (base e)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

def cohort_cube(
    df: pd.DataFrame,
    by: List[str],
    min_n: int = 120,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna:
      1) summary_cohort: 1 linha por coorte (n, métricas agregadas, JS vs baseline)
      2) long_numbers: 1 linha por (coorte, number) com count/rate
    """
    if "ano" not in df.columns:
        df = add_time_columns(df)

    # baseline geral
    base_counts, base_n = _counts_for_mask(df, np.ones(len(df), dtype=bool))
    base_rate = _rates(base_counts, base_n)

    # métricas por sorteio para agregação
    feats = pd.DataFrame({
        "sum": df["nums"].apply(sum_nums),
        "odds": df["nums"].apply(count_odds),
        "primes": df["nums"].apply(count_primes),
        "birthdays": df["nums"].apply(count_birthdays),
        "amp": df["nums"].apply(amplitude),
        "has_consec": df["nums"].apply(lambda a: 1 if has_consecutive(a) else 0),
    })

    work = df.copy()
    for c in feats.columns:
        work[c] = feats[c]

    g = work.groupby(by, dropna=False)

    sum_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    for key, grp in g:
        n = len(grp)
        if n < min_n:
            continue

        counts = np.zeros(60, dtype=float)
        arr = grp[DEZENAS].to_numpy(dtype=int)
        for row in arr:
            for x in row:
                counts[int(x)-1] += 1

        rate = _rates(counts, n)

        # JS divergence: usa vetores normalizados como distribuição (não “rate por sorteio”)
        p = rate / rate.sum()
        q = base_rate / base_rate.sum()
        js = js_divergence(p, q)

        # força simples (amostra × divergência)
        strength = float(js * math.log(1 + n))

        # summary
        row = {}
        if isinstance(key, tuple):
            for col, v in zip(by, key):
                row[col] = v
        else:
            row[by[0]] = key

        row.update({
            "n_draws": int(n),
            "mean_sum": float(grp["sum"].mean()),
            "mean_odds": float(grp["odds"].mean()),
            "mean_primes": float(grp["primes"].mean()),
            "mean_birthdays": float(grp["birthdays"].mean()),
            "mean_amp": float(grp["amp"].mean()),
            "p_has_consec": float(grp["has_consec"].mean()),
            "js_vs_all": float(js),
            "strength": float(strength),
        })
        sum_rows.append(row)

        # long numbers
        for num in NUMBERS:
            long_rows.append({
                **{col: row[col] for col in by},
                "number": int(num),
                "count": int(counts[num-1]),
                "rate_per_draw": float(rate[num-1]),
            })

    summary = pd.DataFrame(sum_rows).sort_values(["strength","n_draws"], ascending=False)
    longn = pd.DataFrame(long_rows)
    return summary, longn


# -----------------------------
# Interações (coortes cruzadas)
# -----------------------------
def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # exemplos úteis e controláveis
    df["mes_x_decada"] = df["mes"].astype("string") + "_" + df["decada"].astype("string")
    df["mes_x_semestre"] = df["mes"].astype("string") + "_" + df["semestre"].astype("string")
    df["mes_x_quinzena"] = df["mes"].astype("string") + "_" + df["quinzena"].astype("string")
    df["bim_x_sem"] = df["bimestre"].astype("string") + "_" + df["semestre"].astype("string")
    return df


# -----------------------------
# Prior temporal para um “contexto alvo”
# -----------------------------
@dataclass
class TargetContext:
    date: pd.Timestamp
    ano: int
    mes: int
    dia_mes: int
    semana_mes: int
    quinzena: int
    bimestre: int
    trimestre: int
    semestre: int
    decada: int
    meia_decada: int
    bi_anual: int
    fase_mes: str
    dia15_win2: str
    soma_dr: int  # opcional (numerologia)

def build_context(target_date: pd.Timestamp) -> TargetContext:
    d = pd.to_datetime(target_date)
    ano = int(d.year)
    mes = int(d.month)
    dia = int(d.day)
    semana = week_of_month(d)
    quinzena = 1 if dia <= 15 else 2
    bimestre = (mes - 1)//2 + 1
    trimestre = (mes - 1)//3 + 1
    semestre = (mes - 1)//6 + 1
    decada = (ano // 10) * 10
    meia_dec = (ano // 5) * 5
    bi_anual = (ano // 2) * 2
    fase = day_phase(dia)
    dia15 = day_window(dia, 15, 2)

    # “numerologia”: raiz digital da soma média esperada (só para brincar no contexto)
    # aqui deixo fixo pela data (você pode mudar para outro ritual)
    soma_dr = (dia + mes + ano) % 9
    soma_dr = 9 if soma_dr == 0 else soma_dr

    return TargetContext(
        date=d, ano=ano, mes=mes, dia_mes=dia, semana_mes=semana, quinzena=quinzena,
        bimestre=bimestre, trimestre=trimestre, semestre=semestre,
        decada=decada, meia_decada=meia_dec, bi_anual=bi_anual,
        fase_mes=fase, dia15_win2=dia15, soma_dr=soma_dr
    )

def _beta_smooth_rate(counts: np.ndarray, n_draws: int, alpha: float = 1.0, beta: float = 9.0) -> np.ndarray:
    # prior média ~0.1 (porque 6/60)
    denom = float(n_draws + alpha + beta)
    return (counts + alpha) / max(denom, 1e-12)

def temporal_prior(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    min_n: int = 160,
    recent_years: int = 12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Gera uma distribuição (60,) de pesos para dezenas condicionada ao período alvo,
    usando vários buckets + interações, com:
      - amostra mínima
      - smoothing beta
      - recência
      - pesos por (strength ≈ JS * log(n))
    """
    if "ano" not in df.columns:
        df = add_time_columns(df)
    df = add_interactions(df)

    ctx = build_context(target_date)
    dsort = pd.to_datetime(df["data_sorteio"], errors="coerce")
    cutoff = ctx.date - pd.DateOffset(years=recent_years)
    mask_recent = (dsort >= cutoff) & dsort.notna()

    # baseline
    base_counts, base_n = _counts_for_mask(df, np.ones(len(df), dtype=bool))
    base_rate = _beta_smooth_rate(base_counts, base_n)

    # buckets principais + interações
    specs = [
        ("mes", ctx.mes),
        ("quinzena", ctx.quinzena),
        ("semana_mes", ctx.semana_mes),
        ("fase_mes", ctx.fase_mes),
        ("bimestre", ctx.bimestre),
        ("trimestre", ctx.trimestre),
        ("semestre", ctx.semestre),
        ("bi_anual", ctx.bi_anual),
        ("meia_decada", ctx.meia_decada),
        ("decada", ctx.decada),
        ("dia15_win2", ctx.dia15_win2),
        # interações (bem nerd, com guarda de amostra)
        ("mes_x_decada", f"{ctx.mes}_{ctx.decada}"),
        ("mes_x_quinzena", f"{ctx.mes}_{ctx.quinzena}"),
        ("bim_x_sem", f"{ctx.bimestre}_{ctx.semestre}"),
    ]

    used = []
    logw = np.zeros(60, dtype=float)

    for col, val in specs:
        if col not in df.columns:
            continue
        m_all = (df[col] == val).to_numpy()
        counts_all, n_all = _counts_for_mask(df, m_all)

        if n_all < min_n:
            continue

        # recência (se tiver amostra razoável)
        m_rec = m_all & mask_recent.to_numpy()
        counts_r, n_r = _counts_for_mask(df, m_rec)

        p_all = _beta_smooth_rate(counts_all, n_all)
        if n_r >= max(50, min_n // 3):
            p_r = _beta_smooth_rate(counts_r, n_r)
            p = 0.65 * p_r + 0.35 * p_all
        else:
            p = p_all

        # força do bucket vs baseline (JS)
        js = js_divergence((p/p.sum()), (base_rate/base_rate.sum()))
        strength = float(js * math.log(1 + n_all))

        # peso do bucket
        w = strength

        logw += w * np.log(np.clip(p, 1e-12, 1.0))

        used.append({"bucket": col, "value": str(val), "n": int(n_all), "n_recent": int(n_r), "js": float(js), "w": float(w)})

    # fallback se nada entrou
    if not used:
        probs = np.ones(60) / 60.0
        return probs, {"fallback": True, "used": []}

    # softmax para virar distribuição
    logw -= logw.max()
    probs = np.exp(logw)
    probs = probs / probs.sum()

    return probs, {"fallback": False, "used": used}


def weighted_sample_6(probs: np.ndarray, rng: random.Random) -> List[int]:
    # amostra sem reposição com pesos
    w = probs.astype(float).copy()
    out = []
    for _ in range(6):
        s = float(w.sum())
        if s <= 0:
            idx = rng.randrange(60)
        else:
            r = rng.random() * s
            acc = 0.0
            idx = 59
            for i in range(60):
                acc += float(w[i])
                if acc >= r:
                    idx = i
                    break
        out.append(idx + 1)
        w[idx] = 0.0
    return sorted(out)