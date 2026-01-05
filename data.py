# data.py
import pandas as pd

COLS = {
    "concurso": "Int64",
    "data_sorteio": "datetime64[ns]",
    "dia_semana": "string",
    "tipo_concurso": "string",
    "status": "string",
    "acumulado": "boolean",
    "local_sorteio": "string",
    "cidade_sorteio": "string",
    "uf_sorteio": "string",
    "d1": "Int64", "d2": "Int64", "d3": "Int64", "d4": "Int64", "d5": "Int64", "d6": "Int64",
    "ganhadores_sena": "Int64",
    "premio_sena": "float64",
    "ganhadores_quina": "Int64",
    "premio_quina": "float64",
    "ganhadores_quadra": "Int64",
    "premio_quadra": "float64",
    "valor_arrecadado": "float64",
    "valor_acumulado_proximo_concurso": "float64",
    "valor_estimado_proximo_concurso": "float64",
    "data_proximo_concurso": "datetime64[ns]",
    "indicador_concurso_especial": "Int64",
}

DEZENAS = ["d1","d2","d3","d4","d5","d6"]

def load_datalake_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")

    # normaliza header (caso venha com espaços)
    df.columns = [c.strip() for c in df.columns]

    # tipagem “segura”
    if "data_sorteio" in df.columns:
        df["data_sorteio"] = pd.to_datetime(df["data_sorteio"], errors="coerce", dayfirst=True)
    if "data_proximo_concurso" in df.columns:
        df["data_proximo_concurso"] = pd.to_datetime(df["data_proximo_concurso"], errors="coerce", dayfirst=True)

    # converte boolean de acumulado se vier como 0/1, "VERDADEIRO/FALSO", etc.
    if "acumulado" in df.columns:
        df["acumulado"] = df["acumulado"].astype("boolean")

    # garante dezenas como ints e ordena dentro de cada concurso
    for c in DEZENAS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # validações mínimas
    _validate(df)

    # cria coluna com lista das dezenas (ordenadas)
    df["nums"] = df[DEZENAS].apply(lambda r: sorted([int(x) for x in r.values if pd.notna(x)]), axis=1)

    # ordena histórico
    df = df.sort_values("concurso").reset_index(drop=True)
    return df


def _validate(df: pd.DataFrame) -> None:
    # concurso único
    if df["concurso"].isna().any():
        raise ValueError("Há concursos sem número (concurso).")

    if df["concurso"].duplicated().any():
        dups = df.loc[df["concurso"].duplicated(), "concurso"].tolist()[:10]
        raise ValueError(f"Concursos duplicados encontrados (exemplos): {dups}")

    # cada linha precisa ter 6 dezenas válidas
    for i, row in df.iterrows():
        nums = [row.get(c) for c in DEZENAS]
        nums = [int(x) for x in nums if pd.notna(x)]
        if len(nums) != 6:
            raise ValueError(f"Concurso {row['concurso']} não tem 6 dezenas válidas.")
        if len(set(nums)) != 6:
            raise ValueError(f"Concurso {row['concurso']} tem dezenas repetidas.")
        if any(n < 1 or n > 60 for n in nums):
            raise ValueError(f"Concurso {row['concurso']} tem dezena fora de 1..60.")
