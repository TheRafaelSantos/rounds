# analysis_temporal.py
# Gera análises temporais e expõe um "perfil temporal" reutilizável pelo rounds.py/model.py.
#
# Uso (análise completa):
#   python analysis_temporal.py --xlsx datalake_megasena.xlsx
#
# Uso (somente exportar perfil do próximo concurso, usando data_proximo_concurso do Excel):
#   python analysis_temporal.py --xlsx datalake_megasena.xlsx --export-profile
#
# Uso (recortar últimos N anos para perfil/relatórios):
#   python analysis_temporal.py --xlsx datalake_megasena.xlsx --recent-years 10
#   python analysis_temporal.py --xlsx datalake_megasena.xlsx --export-profile --recent-years 10
#
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from data import load_datalake_xlsx, DEZENAS

MAX_N = 60
P_IN_DRAW = 6 / 60  # p(dezena aparecer em um concurso)


# ----------------------------
# Buckets temporais
# ----------------------------
def week_of_month(d: pd.Timestamp) -> int:
    # 1..5 em blocos de 7 dias; 0 se inválido
    if pd.isna(d):
        return 0
    return int((int(d.day) - 1) // 7 + 1)


def add_time_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "data_sorteio" not in df.columns:
        raise ValueError("Não achei coluna data_sorteio no df.")

    # garante datetime
    df["data_sorteio"] = pd.to_datetime(df["data_sorteio"], errors="coerce")
    d = df["data_sorteio"]

    df["ano"] = d.dt.year.astype("Int64")
    df["mes"] = d.dt.month.astype("Int64")
    df["dia_mes"] = d.dt.day.astype("Int64")

    df["semana_mes"] = d.apply(week_of_month).astype("Int64")

    # quinzena: 1 (1..15) / 2 (16..fim)
    # (Int64 do pandas = aceita NA; cuidado: numpy.astype("Int64") NÃO funciona)
    df["quinzena"] = (((df["dia_mes"] - 1) // 15) + 1).clip(upper=2).astype("Int64")

    df["bimestre"] = ((df["mes"] - 1) // 2 + 1).astype("Int64")
    df["trimestre"] = d.dt.quarter.astype("Int64")
    df["semestre"] = ((df["mes"] - 1) // 6 + 1).astype("Int64")

    df["bienio"] = ((df["ano"] // 2) * 2).astype("Int64")          # 2024, 2026...
    df["meia_decada"] = ((df["ano"] // 5) * 5).astype("Int64")     # 2020, 2025...
    df["decada"] = ((df["ano"] // 10) * 10).astype("Int64")        # 2020, 2030...

    # fase do mês (string categórica)
    df["fase_mes"] = pd.cut(
        df["dia_mes"].astype("float"),
        bins=[0, 10, 20, 31],
        labels=["inicio(1-10)", "meio(11-20)", "fim(21-31)"],
        include_lowest=True,
    ).astype("string")

    return df


# ----------------------------
# Utilitários de contagem / prob
# ----------------------------
def _counts_for_rows(rows: pd.DataFrame) -> np.ndarray:
    """
    Conta ocorrências de cada dezena (1..60) nas linhas fornecidas.
    Cada linha tem 6 dezenas, então counts soma = 6 * n_linhas (se completo).
    """
    counts = np.zeros(MAX_N, dtype=float)
    if rows is None or len(rows) == 0:
        return counts

    arr = rows[DEZENAS].to_numpy(dtype=float)
    for r in arr:
        for x in r:
            if np.isnan(x):
                continue
            n = int(x)
            if 1 <= n <= MAX_N:
                counts[n - 1] += 1.0
    return counts


def _beta_binom_p(counts: np.ndarray, n_eff: float, alpha: float, beta: float) -> np.ndarray:
    """
    Posterior mean em Beta-Binomial:
      count ~ Binomial(n_eff, p), prior p ~ Beta(alpha, beta)
      E[p|data] = (count + alpha) / (n_eff + alpha + beta)
    """
    denom = float(n_eff + alpha + beta)
    if denom <= 0:
        return np.full(MAX_N, P_IN_DRAW, dtype=float)
    return (counts + float(alpha)) / denom


# ----------------------------
# PERFIL TEMPORAL (motor que o model.py usa)
# ----------------------------
def build_temporal_profile(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    *,
    recent_years: int = 0,
    tau_day: float = 3.0,
    alpha: float = 1.0,
    beta: float = 9.0,
) -> Dict[str, Any]:
    """
    Cria um vetor p_combined[1..60] misturando buckets:
      mes, quinzena, semana_mes, fase_mes, bimestre, trimestre, semestre,
      ano, bienio, meia_decada, decada,
    + componente dia_mes suavizado (gauss/exp por distância em dias).

    Observação: isso NÃO “prova padrão”; é só um gerador de prior/viés acadêmico
    para o laboratório.
    """
    if "data_sorteio" not in df.columns:
        raise ValueError("df precisa ter coluna data_sorteio (datetime).")

    df = df.copy()
    df["data_sorteio"] = pd.to_datetime(df["data_sorteio"], errors="coerce")
    df = df.sort_values("concurso").reset_index(drop=True)
    df = add_time_buckets(df)

    # recorte de recência (opcional)
    if recent_years and int(recent_years) > 0:
        max_year = int(df["ano"].max())
        df = df[df["ano"] >= (max_year - int(recent_years) + 1)].copy()
        df = df.sort_values("concurso").reset_index(drop=True)

    td = pd.to_datetime(target_date)

    target = {
        "ano": int(td.year),
        "mes": int(td.month),
        "dia_mes": int(td.day),
        "semana_mes": week_of_month(td),
        "quinzena": 1 if td.day <= 15 else 2,
        "bimestre": int((td.month - 1) // 2 + 1),
        "trimestre": int(((td.month - 1) // 3 + 1)),
        "semestre": int((td.month - 1) // 6 + 1),
        "fase_mes": ("inicio(1-10)" if td.day <= 10 else ("meio(11-20)" if td.day <= 20 else "fim(21-31)")),
        "bienio": int((td.year // 2) * 2),
        "meia_decada": int((td.year // 5) * 5),
        "decada": int((td.year // 10) * 10),
    }

    bucket_cols = [
        "mes",
        "quinzena",
        "semana_mes",
        "fase_mes",
        "bimestre",
        "trimestre",
        "semestre",
        "ano",
        "bienio",
        "meia_decada",
        "decada",
    ]

    by_bucket: Dict[str, Dict[str, Any]] = {}

    logp = np.zeros(MAX_N, dtype=float)
    wsum = 0.0

    def add_component(name: str, p_vec: np.ndarray, n_eff: float) -> None:
        nonlocal logp, wsum
        # peso cresce com n_eff, mas satura em 1.0
        w = float(min(1.0, math.sqrt(max(float(n_eff), 1.0) / 250.0)))
        by_bucket[name] = {
            "n": float(n_eff),
            "w": w,
            "target": target.get(name, None),
            "p": p_vec.tolist(),
        }
        logp += w * np.log(np.maximum(p_vec, 1e-12))
        wsum += w

    # buckets exatos (mes, quinzena, etc.)
    for b in bucket_cols:
        if b not in df.columns:
            continue
        tv = target[b]

        # comparar como string para evitar pegadinhas de dtype (Int64 vs string)
        mask = (df[b].astype("string") == str(tv))
        rows = df[mask]
        n_draws = float(len(rows))

        counts = _counts_for_rows(rows) if n_draws > 0 else np.zeros(MAX_N, dtype=float)
        p_vec = _beta_binom_p(counts, n_draws, alpha, beta)
        add_component(b, p_vec, n_draws)

    # dia do mês suavizado (aproxima “o mais próximo possível”)
    if "dia_mes" in df.columns:
        D = int(target["dia_mes"])
        days = df["dia_mes"].to_numpy(dtype=float)

        # kernel exponencial por distância em dias
        tau = float(max(float(tau_day), 1e-6))
        weights = np.exp(-np.abs(days - float(D)) / tau)
        weights = np.where(np.isnan(days), 0.0, weights)

        n_eff = float(weights.sum())

        counts_w = np.zeros(MAX_N, dtype=float)
        arr = df[DEZENAS].to_numpy(dtype=float)

        for w, r in zip(weights, arr):
            if w <= 0:
                continue
            for x in r:
                if np.isnan(x):
                    continue
                n = int(x)
                if 1 <= n <= MAX_N:
                    counts_w[n - 1] += float(w)

        p_day = _beta_binom_p(counts_w, n_eff, alpha, beta)
        add_component("dia_mes_suave", p_day, n_eff)

    if wsum <= 0:
        p_combined = np.full(MAX_N, P_IN_DRAW, dtype=float)
    else:
        p_combined = np.exp(logp / wsum)

    p_combined = np.clip(p_combined, 1e-6, 0.999999)

    return {
        "target_date": td.date().isoformat(),
        "p_combined": p_combined.tolist(),
        "meta": {
            "alpha": float(alpha),
            "beta": float(beta),
            "tau_day": float(tau_day),
            "recent_years": int(recent_years),
            "expected_p_in_draw": float(P_IN_DRAW),
            "target": target,
        },
        "by_bucket": by_bucket,
    }


def temporal_score(nums, p_combined: np.ndarray) -> float:
    """
    Score médio (log) das 6 dezenas segundo p_combined.
    Menos negativo = melhor.
    """
    logs = [math.log(max(float(p_combined[int(n) - 1]), 1e-12)) for n in nums]
    return float(sum(logs) / len(logs))


# ----------------------------
# Relatórios CSV (desvios / holdout)
# ----------------------------
def to_long(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    cols = ["concurso", "data_sorteio", bucket_col] + DEZENAS
    tmp = df[cols].copy()
    long = tmp.melt(
        id_vars=["concurso", "data_sorteio", bucket_col],
        value_vars=DEZENAS,
        value_name="dezena",
    )
    long["dezena"] = pd.to_numeric(long["dezena"], errors="coerce").astype("Int64")
    # bucket pode ser Int64/string etc.
    long[bucket_col] = long[bucket_col].astype("string")
    return long.dropna(subset=[bucket_col, "dezena"])


def normal_pvalue_from_z(z: np.ndarray) -> np.ndarray:
    # p-valor bicaudal via erfc
    return np.vectorize(lambda x: math.erfc(abs(float(x)) / math.sqrt(2)))(z)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    # Benjamini–Hochberg (q-values)
    p = pvals.copy()
    n = int(p.size)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = float(ranked[i]) * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty_like(q)
    out[order] = q
    return out


def bucket_number_deviation(df: pd.DataFrame, bucket_col: str, outdir: str) -> None:
    """
    Para cada bucket (ex.: mês=1..12), calcula por dezena:
      - contagem
      - esperado = n_bucket * 0.1
      - z-score binomial (aprox)
      - p-value e q-value (BH-FDR)
    """
    os.makedirs(outdir, exist_ok=True)

    long = to_long(df, bucket_col)
    tab = long.groupby([bucket_col, "dezena"]).size().unstack(fill_value=0)

    # garante colunas 1..60
    for n in range(1, MAX_N + 1):
        if n not in tab.columns:
            tab[n] = 0
    tab = tab[sorted(tab.columns)]

    # número de concursos por bucket
    n_conc = df.dropna(subset=[bucket_col]).groupby(df[bucket_col].astype("string"))["concurso"].count()
    n_conc = n_conc.reindex(tab.index)

    rows = []
    for b in tab.index:
        k = int(n_conc.loc[b]) if b in n_conc.index and not pd.isna(n_conc.loc[b]) else 0
        exp = k * P_IN_DRAW
        var = k * P_IN_DRAW * (1 - P_IN_DRAW)
        sd = math.sqrt(var) if var > 0 else 1.0

        counts = tab.loc[b].values.astype(float)
        z = (counts - exp) / sd
        p = normal_pvalue_from_z(z)
        q = bh_fdr(p)

        for dezena, c, zz, pp, qq in zip(tab.columns, counts, z, p, q):
            rows.append({
                bucket_col: b,
                "n_concursos": k,
                "dezena": int(dezena),
                "count": int(c),
                "expected": float(exp),
                "z": float(zz),
                "p": float(pp),
                "q_fdr": float(qq),
            })

    out = pd.DataFrame(rows).sort_values([bucket_col, "q_fdr", "p"])
    out.to_csv(os.path.join(outdir, f"deviation_by_{bucket_col}.csv"), index=False, encoding="utf-8")

    # Top 10 mais fora por bucket
    top10 = out.groupby(bucket_col, sort=False).head(10)
    top10.to_csv(os.path.join(outdir, f"top10_deviation_by_{bucket_col}.csv"), index=False, encoding="utf-8")


def seasonality_holdout_score(
    df: pd.DataFrame,
    bucket_col: str,
    outdir: str,
    *,
    holdout_ratio: float = 0.2,
    alpha: float = 1.0,
    beta: float = 9.0,
) -> None:
    """
    Valida “sazonalidade”: aprende p(dezena|bucket) no passado e testa no futuro.
    Score = média do log(p) das 6 dezenas reais.
    Baseline = log(0.1).
    """
    os.makedirs(outdir, exist_ok=True)

    df2 = df.dropna(subset=[bucket_col]).copy()
    df2 = df2.sort_values("concurso").reset_index(drop=True)
    if len(df2) < 50:
        # pouco dado -> ignora
        return

    n = len(df2)
    cut = int(round(n * (1 - float(holdout_ratio))))
    cut = max(1, min(cut, n - 1))

    train = df2.iloc[:cut].copy()
    test = df2.iloc[cut:].copy()

    long_tr = to_long(train, bucket_col)
    tab = long_tr.groupby([bucket_col, "dezena"]).size().unstack(fill_value=0)

    for d in range(1, MAX_N + 1):
        if d not in tab.columns:
            tab[d] = 0
    tab = tab[sorted(tab.columns)]

    k_by_bucket = train.groupby(train[bucket_col].astype("string"))["concurso"].count().reindex(tab.index)

    probs: Dict[str, np.ndarray] = {}
    for b in tab.index:
        k = int(k_by_bucket.loc[b]) if b in k_by_bucket.index and not pd.isna(k_by_bucket.loc[b]) else 0
        counts = tab.loc[b].values.astype(float)
        p_hat = (counts + alpha) / (k + alpha + beta)
        probs[str(b)] = p_hat

    def score_row(row) -> float:
        b = str(row[bucket_col])
        if b not in probs:
            return float("nan")
        p_hat = probs[b]
        nums = sorted([int(row[c]) for c in DEZENAS])
        logs = [math.log(max(float(p_hat[n - 1]), 1e-12)) for n in nums]
        return float(np.mean(logs))

    test_scores = test.apply(score_row, axis=1)
    baseline = math.log(P_IN_DRAW)
    delta = test_scores - baseline

    summary = pd.DataFrame({
        "bucket_col": [bucket_col],
        "n_train": [len(train)],
        "n_test": [len(test)],
        "baseline_logp": [baseline],
        "mean_test_logp": [float(np.nanmean(test_scores))],
        "mean_delta_vs_uniform": [float(np.nanmean(delta))],
    })
    summary.to_csv(os.path.join(outdir, f"seasonality_score_{bucket_col}.csv"), index=False, encoding="utf-8")


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--outdir", default=os.path.join("outputs", "temporal"))
    ap.add_argument("--recent-years", type=int, default=0)
    ap.add_argument("--export-profile", action="store_true")
    args = ap.parse_args()

    df = load_datalake_xlsx(args.xlsx).sort_values("concurso").reset_index(drop=True)

    # exportar profile do próximo concurso (usando data_proximo_concurso do Excel)
    if args.export_profile:
        if "data_proximo_concurso" not in df.columns or df["data_proximo_concurso"].isna().all():
            raise ValueError("Não achei data_proximo_concurso no Excel para exportar o profile.")
        target_date = pd.to_datetime(df["data_proximo_concurso"].iloc[-1], errors="coerce")
        if pd.isna(target_date):
            raise ValueError("data_proximo_concurso inválida (não deu para converter em data).")

        prof = build_temporal_profile(df, target_date, recent_years=int(args.recent_years))
        os.makedirs(args.outdir, exist_ok=True)
        out_path = os.path.join(args.outdir, "profile_temporal.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(prof, f, ensure_ascii=False, indent=2)
        print(f"Profile exportado: {out_path}")
        return

    # análise completa / CSVs
    df = add_time_buckets(df)

    if args.recent_years and int(args.recent_years) > 0:
        max_year = int(df["ano"].max())
        df = df[df["ano"] >= (max_year - int(args.recent_years) + 1)].copy()
        df = df.sort_values("concurso").reset_index(drop=True)

    buckets = [
        "dia_mes",
        "semana_mes",
        "quinzena",
        "fase_mes",
        "mes",
        "bimestre",
        "trimestre",
        "semestre",
        "ano",
        "bienio",
        "meia_decada",
        "decada",
    ]

    print(f"Concursos analisados: {len(df)} | {int(df['concurso'].min())}..{int(df['concurso'].max())}")

    for b in buckets:
        if b not in df.columns:
            continue
        print(f"[+] Gerando desvios por {b} ...")
        bucket_number_deviation(df, b, args.outdir)

        # valida sazonalidade só em buckets “grandes”
        if b in {"fase_mes", "mes", "bimestre", "trimestre", "semestre", "ano", "decada"}:
            print(f"    Validando sazonalidade (holdout) para {b} ...")
            seasonality_holdout_score(df, b, args.outdir)

    print(f"OK. Arquivos em: {args.outdir}")


if __name__ == "__main__":
    main()