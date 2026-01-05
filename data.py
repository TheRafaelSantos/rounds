# data.py
from __future__ import annotations

import pandas as pd

DEZENAS = ["d1", "d2", "d3", "d4", "d5", "d6"]

def load_datalake_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # concurso
    if "concurso" in df.columns:
        df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce").astype("Int64")
    else:
        df["concurso"] = pd.Series(range(1, len(df) + 1), dtype="Int64")

    # datas
    for c in ["data_sorteio", "data_proximo_concurso"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # dezenas
    missing = [c for c in DEZENAS if c not in df.columns]
    if missing:
        raise ValueError(f"Excel sem colunas {missing}. Esperado: {DEZENAS}")

    for c in DEZENAS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["nums"] = df[DEZENAS].apply(lambda r: sorted(int(x) for x in r.tolist()), axis=1)
    return df