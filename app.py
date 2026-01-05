# rounds.py
# Mega-Sena Lab (acadêmico) - Runner/CLI
# ------------------------------------------------------------
# Requisitos:
#   pip install pandas openpyxl
#
# Estrutura esperada:
#   data.py  -> função load_datalake_xlsx(path) retornando df com colunas:
#              concurso, d1..d6 (e idealmente df["nums"] = lista ordenada)
#   model.py -> class ProfileModel(df, window_recent=...)
#              método suggest(last_nums, n_samples, top_k, seed) -> lista
#              contendo itens no formato: (score, nums, feats) OU dicts.
#
# Este runner:
#  - carrega o datalake do Excel
#  - gera Top-K previsões
#  - salva CSV e logs para placar
#  - avalia quando você passar o resultado real (--result)
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Imports do seu projeto
try:
    from data import load_datalake_xlsx  # type: ignore
except Exception as e:
    raise ImportError(
        "Não consegui importar load_datalake_xlsx de data.py. "
        "Confirme que data.py existe e contém load_datalake_xlsx(path)."
    ) from e

try:
    from model import ProfileModel  # type: ignore
except Exception as e:
    raise ImportError(
        "Não consegui importar ProfileModel de model.py. "
        "Confirme que model.py existe e contém a classe ProfileModel."
    ) from e


DEZENAS_COLS = ["d1", "d2", "d3", "d4", "d5", "d6"]


# -------------------------
# Utilidades
# -------------------------
def parse_nums(s: str) -> List[int]:
    """
    Aceita '1,2,3,4,5,6' ou '01 02 03 04 05 06' etc.
    """
    if not s:
        return []
    parts = []
    for token in s.replace(";", ",").replace("|", ",").replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"Token inválido em --result: '{token}'")
        parts.append(int(token))
    nums = sorted(parts)
    if len(nums) != 6:
        raise ValueError(f"--result precisa ter 6 dezenas; veio {len(nums)} ({nums})")
    if len(set(nums)) != 6:
        raise ValueError(f"--result tem dezenas repetidas: {nums}")
    if any(n < 1 or n > 60 for n in nums):
        raise ValueError(f"--result fora de 1..60: {nums}")
    return nums


def fmt_nums(nums: Sequence[int]) -> str:
    return ",".join(f"{n:02d}" for n in nums)


def overlap_count(a: Sequence[int], b: Sequence[int]) -> int:
    return len(set(a) & set(b))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_iso() -> str:
    # ISO sem micros para ficar legível
    return dt.datetime.now().replace(microsecond=0).isoformat()


def make_run_id(payload: Dict[str, Any]) -> str:
    """
    Gera um run_id estável (hash curto) baseado no payload do experimento.
    """
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()[:12]
    return h


def normalize_suggest_item(item: Any) -> Tuple[float, List[int], Dict[str, Any]]:
    """
    Normaliza o retorno do model.suggest para um formato único:
    (score: float, nums: [int...], feats: dict)

    Suporta:
    - tuple/list: (score, nums, feats)
    - dict: {"score":..., "nums":[...], ...feats}
    """
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        score = float(item[0])
        nums = list(item[1])
        feats = dict(item[2]) if len(item) >= 3 and isinstance(item[2], dict) else {}
        return score, nums, feats

    if isinstance(item, dict):
        if "nums" not in item or "score" not in item:
            raise ValueError("Item dict do suggest precisa conter 'nums' e 'score'.")
        score = float(item["score"])
        nums = list(item["nums"])
        feats = {k: v for k, v in item.items() if k not in ("score", "nums")}
        return score, nums, feats

    raise ValueError(f"Formato inesperado retornado por suggest(): {type(item)}")


# -------------------------
# Logs
# -------------------------
def append_csv_row(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mega-Sena Lab (acadêmico) - gera previsões e registra placar."
    )
    ap.add_argument("--xlsx", default="datalake_megasena.xlsx", help="Caminho do Excel do datalake.")
    ap.add_argument("--include-special", action="store_true",
                    help="Inclui concursos especiais (Mega da Virada) no histórico e na previsão.")
    ap.add_argument("--window-recent", type=int, default=300, help="Janela recente (ex.: 300).")
    ap.add_argument("--n-samples", type=int, default=200_000, help="Qtd de candidatos aleatórios.")
    ap.add_argument("--top-k", type=int, default=20, help="Qtd de palpites no Top-K.")
    ap.add_argument("--seed", type=int, default=123, help="Seed para reprodutibilidade.")
    ap.add_argument("--w-recent", type=float, default=0.55,
                    help="Peso da janela recente vs histórico (se seu model usar).")
    ap.add_argument("--outdir", default="outputs", help="Pasta para salvar resultados (CSVs/logs).")
    ap.add_argument("--result", default="", help="Resultado real (6 dezenas) para avaliação. Ex: '05,11,28,33,54,55'")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # -------------------------
    # Load datalake
    # -------------------------
    df = load_datalake_xlsx(args.xlsx)

    # Se o seu load_datalake_xlsx já filtra/normaliza, ótimo.
    # Caso não filtre especial, fazemos aqui:
    if (not args.include_special) and ("indicador_concurso_especial" in df.columns):
        df = df[df["indicador_concurso_especial"].fillna(1).astype(int) != 2].copy()

    if "concurso" not in df.columns:
        raise ValueError("Seu Excel precisa ter a coluna 'concurso'.")

    # Garante nums
    if "nums" not in df.columns:
        # tenta montar de d1..d6
        missing = [c for c in DEZENAS_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Não encontrei colunas {missing} e também não existe df['nums'].")
        df["nums"] = df[DEZENAS_COLS].apply(
            lambda r: sorted([int(x) for x in r.values if pd.notna(x)]),
            axis=1
        )

    df = df.sort_values("concurso").reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Histórico muito pequeno no datalake para rodar um laboratório legal.")

    last_row = df.iloc[-1]
    last_concurso = int(last_row["concurso"])
    last_nums = list(last_row["nums"])

    next_concurso = last_concurso + 1

    # -------------------------
    # Build model
    # -------------------------
    model = ProfileModel(df, window_recent=args.window_recent)

    # Alguns modelos aceitam w_recent na score/suggest. Se o seu não aceitar, não tem problema:
    # a função normalize_suggest_item já cuida do retorno; aqui só tentamos chamar do jeito mais útil.
    try:
        raw_best = model.suggest(
            last_nums=last_nums,
            n_samples=args.n_samples,
            top_k=args.top_k,
            seed=args.seed,
            w_recent=args.w_recent,  # pode falhar se o seu método não aceitar
        )
    except TypeError:
        raw_best = model.suggest(
            last_nums=last_nums,
            n_samples=args.n_samples,
            top_k=args.top_k,
            seed=args.seed,
        )

    best: List[Tuple[float, List[int], Dict[str, Any]]] = [normalize_suggest_item(x) for x in raw_best]

    # -------------------------
    # Print summary
    # -------------------------
    print("=" * 88)
    print("MEGA-SENA LAB (acadêmico) — PREVISÕES")
    print("=" * 88)
    print(f"Datalake: {args.xlsx}")
    print(f"Concursos no histórico (após filtros): {len(df)}")
    if "data_sorteio" in df.columns and pd.notna(last_row.get("data_sorteio", None)):
        try:
            d = pd.to_datetime(last_row["data_sorteio"]).date()
            print(f"Último concurso: {last_concurso} ({d}) | dezenas: {fmt_nums(last_nums)}")
        except Exception:
            print(f"Último concurso: {last_concurso} | dezenas: {fmt_nums(last_nums)}")
    else:
        print(f"Último concurso: {last_concurso} | dezenas: {fmt_nums(last_nums)}")

    print(f"Próximo concurso (previsto): {next_concurso}")
    print(f"Amostras: {args.n_samples:,} | Top-K: {args.top_k} | Seed: {args.seed}")
    print("-" * 88)

    # -------------------------
    # Save frozen predictions CSV + append history logs
    # -------------------------
    timestamp = now_iso()
    exp_payload = {
        "timestamp": timestamp,
        "xlsx": os.path.abspath(args.xlsx),
        "include_special": bool(args.include_special),
        "window_recent": int(args.window_recent),
        "n_samples": int(args.n_samples),
        "top_k": int(args.top_k),
        "seed": int(args.seed),
        "w_recent": float(args.w_recent),
        "last_concurso": last_concurso,
        "last_nums": last_nums,
        "next_concurso": next_concurso,
    }
    run_id = make_run_id(exp_payload)

    frozen_csv = os.path.join(args.outdir, f"predicoes_concurso_{next_concurso}_run_{run_id}.csv")
    history_csv = os.path.join(args.outdir, "predictions_history.csv")
    eval_csv = os.path.join(args.outdir, "evaluation_history.csv")

    # monta dataframe das previsões
    rows = []
    for rank, (score, nums, feats) in enumerate(best, start=1):
        row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "rank": rank,
            "next_concurso": next_concurso,
            "last_concurso": last_concurso,
            "last_nums": fmt_nums(last_nums),
            "nums": fmt_nums(nums),
            "score": score,
            "overlap_last": overlap_count(nums, last_nums),
        }
        # coloca feats (se existirem) com prefixo feat_
        for k, v in feats.items():
            row[f"feat_{k}"] = v
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(frozen_csv, index=False, encoding="utf-8")

    # append em predictions_history.csv (log incremental)
    # (monta campo fixo + feats dinâmicos)
    base_fields = [
        "run_id", "timestamp", "rank", "next_concurso", "last_concurso", "last_nums",
        "nums", "score", "overlap_last"
    ]
    feat_fields = sorted([c for c in pred_df.columns if c.startswith("feat_")])
    fields = base_fields + feat_fields

    # garante que todas as colunas existam nos rows
    for r in rows:
        for c in feat_fields:
            r.setdefault(c, "")

    # grava
    if not os.path.exists(history_csv):
        pred_df[fields].to_csv(history_csv, index=False, encoding="utf-8")
    else:
        # append (sem reescrever header)
        with open(history_csv, "a", newline="", encoding="utf-8") as f:
            pred_df[fields].to_csv(f, index=False, header=False)

    # -------------------------
    # Exibe Top-K no terminal
    # -------------------------
    for rank, (score, nums, feats) in enumerate(best, start=1):
        ov = overlap_count(nums, last_nums)
        # imprime algumas features comuns se existirem
        s = feats.get("sum", "")
        odds = feats.get("odds", "")
        primes = feats.get("primes", "")
        birthdays = feats.get("birthdays", "")
        maxrun = feats.get("maxrun", "")

        extra = []
        if s != "": extra.append(f"soma={s}")
        if odds != "": extra.append(f"ímpares={odds}")
        if primes != "": extra.append(f"primos={primes}")
        if birthdays != "": extra.append(f"1-31={birthdays}")
        if maxrun != "": extra.append(f"max_run={maxrun}")

        extra_txt = " | " + " | ".join(extra) if extra else ""
        print(f"{rank:>2}) {fmt_nums(nums)} | overlap_último={ov} | score={score:.6f}{extra_txt}")

    print("-" * 88)
    print(f"CSV congelado: {frozen_csv}")
    print(f"Log previsões:  {history_csv}")

    # -------------------------
    # Avaliação (se o usuário passar o resultado real)
    # -------------------------
    if args.result.strip():
        real = parse_nums(args.result.strip())
        print("=" * 88)
        print(f"AVALIAÇÃO — Resultado informado: {fmt_nums(real)}")
        print("=" * 88)

        eval_fields = [
            "run_id", "timestamp", "next_concurso", "result_nums",
            "rank", "pred_nums", "hits", "overlap_last"
        ]

        for rank, (score, nums, feats) in enumerate(best, start=1):
            hits = overlap_count(nums, real)
            ov_last = overlap_count(nums, last_nums)
            print(f"{rank:>2}) pred={fmt_nums(nums)} | hits={hits} | overlap_último={ov_last} | score={score:.6f}")

            append_csv_row(
                eval_csv,
                eval_fields,
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "next_concurso": next_concurso,
                    "result_nums": fmt_nums(real),
                    "rank": rank,
                    "pred_nums": fmt_nums(nums),
                    "hits": hits,
                    "overlap_last": ov_last,
                },
            )

        print("-" * 88)
        print(f"Log avaliação:  {eval_csv}")

    print("=" * 88)
    print(f"RUN_ID: {run_id}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\n[ERRO] {type(e).__name__}: {e}", file=sys.stderr)
        raise