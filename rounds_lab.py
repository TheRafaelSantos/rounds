# rounds_lab.py

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# -----------------------------
# Helpers: detecção de colunas
# -----------------------------

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _is_dezena_col(series: pd.Series) -> bool:
    # tenta identificar uma coluna que pareça uma "dezena" (1..60)
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if len(s) < 200:  # precisa ter bastante linha pra ser confiável
        return False
    # valores inteiros entre 1 e 60
    ok = (s >= 1) & (s <= 60) & (s % 1 == 0)
    if ok.mean() < 0.98:  # quase tudo precisa estar dentro de 1..60
        return False
    # tem variação suficiente
    if s.nunique() < 40:
        return False
    return True

def detect_dezenas_cols(df: pd.DataFrame) -> List[str]:
    # 1) tenta por nome (bola1..bola6, dezena1..)
    cols = list(df.columns)
    name_hits = []
    for c in cols:
        n = _norm(str(c))
        if any(k in n for k in ["bola", "dezena", "n1", "n2", "n3", "n4", "n5", "n6"]):
            name_hits.append(c)

    # se tiver 6 colunas bem claras, ok
    # mas como nomes variam, vamos validar pelo conteúdo e depois pegar 6
    decade_candidates = [c for c in cols if _is_dezena_col(df[c])]
    # prioriza os que têm "bola/dezena" no nome
    decade_candidates_sorted = sorted(
        decade_candidates,
        key=lambda c: (0 if c in name_hits else 1, str(c))
    )
    if len(decade_candidates_sorted) >= 6:
        return decade_candidates_sorted[:6]

    raise ValueError(
        "Não consegui detectar as 6 colunas de dezenas (1..60). "
        "Verifique se o Excel tem as 6 dezenas em colunas separadas."
    )

def detect_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        n = _norm(str(c))
        if "data" in n:
            return c
    # fallback: tenta achar uma coluna com muitos valores parseáveis como data
    best = None
    best_score = 0
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        score = s.notna().mean()
        if score > best_score and score > 0.7:
            best = c
            best_score = score
    return best

def detect_uf_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        n = _norm(str(c))
        if n in ["uf", "estado"] or " uf" in n or n.endswith("uf"):
            return c
    return None

def detect_city_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        n = _norm(str(c))
        if any(k in n for k in ["cidade", "municip", "local", "município", "municipio"]):
            return c
    return None

def detect_concurso_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        n = _norm(str(c))
        if any(k in n for k in ["concurso", "numero", "número", "round"]):
            # evita pegar colunas "numeroSorteados" etc, mas se for, ok.
            return c
    return None

def detect_acumulou_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        n = _norm(str(c))
        if "acumul" in n:
            return c
    return None


# -----------------------------
# Cálculos principais
# -----------------------------

@dataclass
class KeyStats:
    n_sorteios: int
    first_round: Optional[int]
    last_round: Optional[int]
    first_date: Optional[str]
    last_date: Optional[str]

def compute_key_stats(df: pd.DataFrame, concurso_col: Optional[str], date_col: Optional[str]) -> KeyStats:
    n = len(df)
    first_round = last_round = None
    first_date = last_date = None

    if concurso_col and concurso_col in df.columns:
        cc = pd.to_numeric(df[concurso_col], errors="coerce").dropna()
        if len(cc):
            first_round = int(cc.min())
            last_round = int(cc.max())

    if date_col and date_col in df.columns:
        dd = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dropna()
        if len(dd):
            first_date = dd.min().date().isoformat()
            last_date = dd.max().date().isoformat()

    return KeyStats(n_sorteios=n, first_round=first_round, last_round=last_round, first_date=first_date, last_date=last_date)

def flatten_dezenas(df: pd.DataFrame, dezenas_cols: List[str]) -> pd.Series:
    vals = pd.to_numeric(df[dezenas_cols].stack(), errors="coerce").dropna().astype(int)
    return vals

def freq_overall(df: pd.DataFrame, dezenas_cols: List[str]) -> pd.DataFrame:
    s = flatten_dezenas(df, dezenas_cols)
    freq = s.value_counts().sort_index()
    out = pd.DataFrame({"dezena": freq.index, "freq": freq.values})
    out["pct"] = out["freq"] / out["freq"].sum()
    return out

def odd_even_distribution(df: pd.DataFrame, dezenas_cols: List[str]) -> pd.DataFrame:
    d = df[dezenas_cols].apply(pd.to_numeric, errors="coerce")
    odd_counts = (d % 2 == 1).sum(axis=1)
    dist = odd_counts.value_counts().sort_index()
    return pd.DataFrame({"qtd_impares": dist.index, "freq_sorteios": dist.values, "pct": dist.values / dist.values.sum()})

def sum_distribution(df: pd.DataFrame, dezenas_cols: List[str]) -> pd.DataFrame:
    d = df[dezenas_cols].apply(pd.to_numeric, errors="coerce")
    sums = d.sum(axis=1)
    # hist em bins de 10
    bins = list(range(0, 361, 10))
    cat = pd.cut(sums, bins=bins, include_lowest=True, right=False)
    dist = cat.value_counts().sort_index()
    out = pd.DataFrame({"faixa_soma": dist.index.astype(str), "freq_sorteios": dist.values})
    out["pct"] = out["freq_sorteios"] / out["freq_sorteios"].sum()
    return out

def range_bands(df: pd.DataFrame, dezenas_cols: List[str]) -> pd.DataFrame:
    d = df[dezenas_cols].apply(pd.to_numeric, errors="coerce")
    b1 = ((d >= 1) & (d <= 20)).sum(axis=1)
    b2 = ((d >= 21) & (d <= 40)).sum(axis=1)
    b3 = ((d >= 41) & (d <= 60)).sum(axis=1)
    key = list(zip(b1.astype(int), b2.astype(int), b3.astype(int)))
    dist = pd.Series(key).value_counts().sort_index()
    out = pd.DataFrame({"faixas_(1-20,21-40,41-60)": [str(k) for k in dist.index], "freq_sorteios": dist.values})
    out["pct"] = out["freq_sorteios"] / out["freq_sorteios"].sum()
    return out

def overlaps_with_previous(df: pd.DataFrame, dezenas_cols: List[str]) -> pd.DataFrame:
    d = df[dezenas_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")
    overlaps = []
    prev = None
    for _, row in d.iterrows():
        cur = set(int(x) for x in row.dropna().tolist())
        if prev is None:
            overlaps.append(None)
        else:
            overlaps.append(len(cur & prev))
        prev = cur
    s = pd.Series(overlaps).dropna().astype(int)
    dist = s.value_counts().sort_index()
    out = pd.DataFrame({"overlap_qtd_com_anterior": dist.index, "freq_sorteios": dist.values})
    out["pct"] = out["freq_sorteios"] / out["freq_sorteios"].sum()
    return out

def freq_by_year(df: pd.DataFrame, dezenas_cols: List[str], date_col: Optional[str]) -> Optional[pd.DataFrame]:
    if not date_col or date_col not in df.columns:
        return None
    dd = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    tmp = df.copy()
    tmp["_year"] = dd.dt.year
    tmp = tmp.dropna(subset=["_year"])
    tmp["_year"] = tmp["_year"].astype(int)

    rows = []
    for y, g in tmp.groupby("_year"):
        s = flatten_dezenas(g, dezenas_cols)
        freq = s.value_counts().sort_index()
        row = {"year": y}
        for n in range(1, 61):
            row[str(n)] = int(freq.get(n, 0))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("year")
    return out

def by_location(df: pd.DataFrame, dezenas_cols: List[str], city_col: Optional[str], uf_col: Optional[str], acumulou_col: Optional[str]) -> Optional[pd.DataFrame]:
    if not city_col and not uf_col:
        return None

    tmp = df.copy()

    if city_col and city_col in tmp.columns:
        tmp["_cidade"] = tmp[city_col].astype(str).str.strip()
    else:
        tmp["_cidade"] = "N/A"

    if uf_col and uf_col in tmp.columns:
        tmp["_uf"] = tmp[uf_col].astype(str).str.strip().str.upper()
    else:
        tmp["_uf"] = "N/A"

    # acumulou (se existir)
    if acumulou_col and acumulou_col in tmp.columns:
        a = tmp[acumulou_col].astype(str).str.lower().str.strip()
        # aceita "sim/nao", "true/false", "acumulou"
        tmp["_acumulou"] = a.isin(["sim", "s", "true", "1", "acumulou", "acumulada", "yes"])
    else:
        tmp["_acumulou"] = False

    # métricas por UF e por cidade
    group_cols = ["_uf", "_cidade"] if city_col else ["_uf"]
    rows = []
    for keys, g in tmp.groupby(group_cols):
        if isinstance(keys, tuple):
            uf, cidade = keys
        else:
            uf, cidade = keys, "N/A"

        s = flatten_dezenas(g, dezenas_cols)
        freq = s.value_counts()
        rows.append({
            "uf": uf,
            "cidade": cidade,
            "sorteios": int(len(g)),
            "acumulou_qtd": int(g["_acumulou"].sum()),
            "acumulou_pct": float(g["_acumulou"].mean()) if len(g) else 0.0,
            "dezena_mais_frequente": int(freq.idxmax()) if len(freq) else None,
            "freq_da_mais_frequente": int(freq.max()) if len(freq) else None,
        })

    out = pd.DataFrame(rows).sort_values(["sorteios", "acumulou_pct"], ascending=[False, False])
    return out


# -----------------------------
# IO: leitura e relatório
# -----------------------------

def read_xlsx(path: str) -> pd.DataFrame:
    # lê a 1ª planilha por padrão (robusto)
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    return df

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_markdown(report_path: Path, md: str) -> None:
    report_path.write_text(md, encoding="utf-8")

def build_markdown_report(
    ks: KeyStats,
    freq: pd.DataFrame,
    odd_even: pd.DataFrame,
    sums: pd.DataFrame,
    bands: pd.DataFrame,
    overlaps: pd.DataFrame,
    loc: Optional[pd.DataFrame],
) -> str:
    def head_table(df: pd.DataFrame, n=12) -> str:
        return df.head(n).to_markdown(index=False)

    md = []
    md.append("# Rounds Lab — Dashboard do Data Lake\n")
    md.append(f"- Sorteios: **{ks.n_sorteios}**")
    if ks.first_round is not None and ks.last_round is not None:
        md.append(f"- Round: **{ks.first_round} → {ks.last_round}**")
    if ks.first_date and ks.last_date:
        md.append(f"- Datas: **{ks.first_date} → {ks.last_date}**")
    md.append("")

    md.append("## Frequência geral (Top 12)\n")
    md.append(head_table(freq.sort_values("freq", ascending=False), 12))
    md.append("")

    md.append("## Distribuição de ímpares por sorteio\n")
    md.append(head_table(odd_even, 10))
    md.append("")

    md.append("## Distribuição da soma (bins de 10)\n")
    md.append(head_table(sums, 15))
    md.append("")

    md.append("## Distribuição por faixas (1-20, 21-40, 41-60)\n")
    md.append(head_table(bands, 15))
    md.append("")

    md.append("## Overlap com o sorteio anterior\n")
    md.append(head_table(overlaps, 10))
    md.append("")

    if loc is not None and len(loc):
        md.append("## Locais (Top 20 por sorteios)\n")
        md.append(head_table(loc, 20))
        md.append("")

    md.append("\n> Observação: este relatório é descritivo/estatístico (não é previsão).")
    return "\n".join(md)


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Caminho do datalake .xlsx")
    ap.add_argument("--dashboard", action="store_true", help="Gera estatísticas e arquivos em artifacts/")
    ap.add_argument("--save_md", action="store_true", help="Salva relatório markdown em artifacts/report.md")
    ap.add_argument("--artifacts", default="artifacts", help="Pasta de saída")
    args = ap.parse_args()

    xlsx_path = args.xlsx
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {xlsx_path}")

    df = read_xlsx(xlsx_path)

    dezenas_cols = detect_dezenas_cols(df)
    date_col = detect_date_col(df)
    uf_col = detect_uf_col(df)
    city_col = detect_city_col(df)
    concurso_col = detect_concurso_col(df)
    acumulou_col = detect_acumulou_col(df)

    ks = compute_key_stats(df, concurso_col, date_col)

    freq = freq_overall(df, dezenas_cols)
    odd_even = odd_even_distribution(df, dezenas_cols)
    sums = sum_distribution(df, dezenas_cols)
    bands = range_bands(df, dezenas_cols)
    overlaps = overlaps_with_previous(df, dezenas_cols)

    loc = by_location(df, dezenas_cols, city_col, uf_col, acumulou_col)

    artifacts_dir = Path(args.artifacts)
    ensure_dir(artifacts_dir)

    # Sempre salva alguns CSVs úteis
    freq.sort_values("dezena").to_csv(artifacts_dir / "freq_overall.csv", index=False, encoding="utf-8")
    odd_even.to_csv(artifacts_dir / "odd_even.csv", index=False, encoding="utf-8")
    sums.to_csv(artifacts_dir / "sum_distribution.csv", index=False, encoding="utf-8")
    bands.to_csv(artifacts_dir / "bands_1_20_21_40_41_60.csv", index=False, encoding="utf-8")
    overlaps.to_csv(artifacts_dir / "overlap_prev.csv", index=False, encoding="utf-8")
    if loc is not None:
        loc.to_csv(artifacts_dir / "by_location.csv", index=False, encoding="utf-8")

    fy = freq_by_year(df, dezenas_cols, date_col)
    if fy is not None:
        fy.to_csv(artifacts_dir / "freq_by_year.csv", index=False, encoding="utf-8")

    md = build_markdown_report(ks, freq, odd_even, sums, bands, overlaps, loc)

    # Print resumo curto no terminal
    print(f"Linhas no datalake: {ks.n_sorteios}")
    if ks.first_round is not None and ks.last_round is not None:
        print(f"Rounds: {ks.first_round} → {ks.last_round}")
    if ks.first_date and ks.last_date:
        print(f"Datas: {ks.first_date} → {ks.last_date}")
    print(f"Colunas dezenas detectadas: {dezenas_cols}")
    if date_col:
        print(f"Coluna de data: {date_col}")
    if uf_col or city_col:
        print(f"Localização: cidade={city_col} | uf={uf_col}")
    if acumulou_col:
        print(f"Coluna acumulou: {acumulou_col}")
    print(f"Arquivos gerados em: {artifacts_dir.resolve()}")

    if args.save_md:
        save_markdown(artifacts_dir / "report.md", md)
        print("Relatório markdown salvo: artifacts/report.md")

if __name__ == "__main__":
    main()
