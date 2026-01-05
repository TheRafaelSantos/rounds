# rounds.py
# Mega-Sena Lab (acadêmico) - Runner/CLI

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from typing import Any, Dict, List, Sequence

import pandas as pd

from data import load_datalake_xlsx
from model import ProfileModel


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_nums(text: str, *, pick: int = 6) -> List[int]:
    import re
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if len(nums) != pick:
        raise ValueError(f"Esperado {pick} dezenas em --result, veio {len(nums)}: {nums}")
    if len(set(nums)) != pick:
        raise ValueError(f"--result tem repetição: {nums}")
    if any(n < 1 or n > 60 for n in nums):
        raise ValueError(f"--result fora de 1..60: {nums}")
    return sorted(nums)


def normalize_nums(nums: Sequence[int]) -> List[int]:
    return sorted(int(x) for x in nums)


def run_id_from_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:10]


def compute_hits(pred: Sequence[int], real: Sequence[int]) -> int:
    return len(set(pred) & set(real))


def _safe_union_fieldnames(path: str, new_fields: List[str]) -> List[str]:
    if not os.path.exists(path):
        return new_fields
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except Exception:
        header = []
    if not header:
        return new_fields
    return list(dict.fromkeys(list(header) + list(new_fields)))


def _rewrite_csv_with_union(path: str, union_fields: List[str]) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        old_fields = reader.fieldnames or []
    if old_fields == union_fields:
        return
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=union_fields)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in union_fields}
            w.writerow(out)
    os.replace(tmp_path, path)


def _row_for_csv(run_id: str, ts: str, d: Dict[str, Any]) -> Dict[str, Any]:
    nums = d["nums"]
    nums_s = " ".join(f"{n:02d}" for n in nums)
    return {
        "run_id": run_id,
        "ts": ts,
        "bucket": d.get("bucket"),
        "bucket_rank": d.get("bucket_rank"),
        "nums": nums_s,
        "typical_raw": d.get("typical_raw"),
        "strategy_score": d.get("strategy_score"),
        "overlap_last": d.get("overlap_last"),
        "odds": d.get("feat_odds"),
        "primes": d.get("feat_primes"),
        "birthdays": d.get("feat_birthdays"),
        "sum": d.get("feat_sum"),
        "decades": d.get("feat_decades"),
        "maxrun": d.get("feat_maxrun"),
        "amp": d.get("feat_amp"),
        "digits": d.get("feat_digits"),
        "hot_raw": d.get("hot_raw"),
        "cold_raw": d.get("cold_raw"),
        "temporal_raw": d.get("temporal_raw"),
        "z_typ": d.get("z_typ"),
        "z_hot": d.get("z_hot"),
        "z_cold": d.get("z_cold"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Mega-Sena Lab - gerar previsões e registrar rodada")
    p.add_argument("--xlsx", default="datalake_megasena.xlsx")
    p.add_argument("--outdir", default="outputs")

    p.add_argument("--portfolio", choices=["mixed12", "plain"], default="mixed12")
    p.add_argument("--top-k", type=int, default=12)

    p.add_argument("--n-samples", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--w-recent", type=float, default=0.55)
    p.add_argument("--window-recent", type=int, default=300)

    p.add_argument("--max-overlap-picks", type=int, default=2)
    p.add_argument("--include-special", action="store_true")
    p.add_argument("--next-is-special", action="store_true")

    p.add_argument("--result", default=None)

    p.add_argument("--temporal-plus", action="store_true")
    p.add_argument("--target-date", default=None)
    p.add_argument("--temporal-recent-years", type=int, default=0)

    args = p.parse_args()

    ensure_dir(args.outdir)

    df = load_datalake_xlsx(args.xlsx).sort_values("concurso").reset_index(drop=True)

    if "indicador_concurso_especial" in df.columns:
        last_is_special = int(df["indicador_concurso_especial"].fillna(0).iloc[-1]) == 2
        if last_is_special and (not args.include_special):
            last_conc = int(df["concurso"].iloc[-1])
            last_date = df["data_sorteio"].iloc[-1] if "data_sorteio" in df.columns else None
            print("[AVISO] Seu último concurso no Excel é ESPECIAL (indicador_concurso_especial=2).")
            print(f"        Concurso {last_conc} ({last_date.date() if hasattr(last_date,'date') else last_date}) será EXCLUÍDO porque --include-special está desligado.")
            print("        Se você quiser incluir, rode com: --include-special\n")

    if (not args.include_special) and ("indicador_concurso_especial" in df.columns):
        df = df[df["indicador_concurso_especial"].fillna(0).astype(int) != 2].copy()
        df = df.sort_values("concurso").reset_index(drop=True)

    last_nums = normalize_nums(df["nums"].iloc[-1])
    last_concurso = int(df["concurso"].iloc[-1]) if "concurso" in df.columns else len(df)

    print(f"Concursos carregados: {len(df)} | Último concurso: {last_concurso} -> {last_nums}")

    model = ProfileModel(df, window_recent=args.window_recent)

    now = dt.datetime.now()
    payload = {
        "ts": now.isoformat(timespec="seconds"),
        "xlsx": os.path.basename(args.xlsx),
        "portfolio": args.portfolio,
        "top_k": int(args.top_k),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "w_recent": float(args.w_recent),
        "window_recent": int(args.window_recent),
        "max_overlap_picks": int(args.max_overlap_picks),
        "include_special": bool(args.include_special),
        "last_concurso": last_concurso,
        "last_nums": last_nums,
        "temporal_plus": bool(args.temporal_plus),
        "target_date": args.target_date,
        "temporal_recent_years": int(args.temporal_recent_years),
        "next_is_special": bool(args.next_is_special),
    }
    rid = run_id_from_payload(payload)

    predictions: List[Dict[str, Any]] = []

    if args.portfolio == "mixed12":
        preds = model.suggest_portfolio_mixed12(
            last_nums=last_nums,
            n_samples=args.n_samples,
            seed=args.seed,
            w_recent=args.w_recent,
            max_overlap_between_picks=args.max_overlap_picks,
        )
        for d in preds:
            predictions.append(_row_for_csv(rid, payload["ts"], d))

        if args.temporal_plus:
            target_date = None
            if args.target_date:
                target_date = pd.to_datetime(args.target_date)
            elif "data_proximo_concurso" in df.columns and (not df["data_proximo_concurso"].isna().all()):
                target_date = pd.to_datetime(df["data_proximo_concurso"].iloc[-1])

            if target_date is None or pd.isna(target_date):
                print("\n[AVISO] temporal_plus ligado, mas não encontrei target_date.")
                print("        Passe --target-date YYYY-MM-DD ou garanta data_proximo_concurso no Excel.")
            else:
                already_picks = [d["nums"] for d in preds]
                plus_rows, prof = model.suggest_temporal_plus3(
                    last_nums=last_nums,
                    target_date=target_date,
                    n_samples=args.n_samples,
                    seed=args.seed + 7,
                    w_recent=args.w_recent,
                    max_overlap_between_picks=args.max_overlap_picks,
                    already_picks=already_picks,
                    recent_years_for_temporal=args.temporal_recent_years,
                    next_is_special=args.next_is_special,
                )
                for d in plus_rows:
                    predictions.append(_row_for_csv(rid, payload["ts"], d))

                tdir = os.path.join(args.outdir, "temporal")
                ensure_dir(tdir)
                with open(os.path.join(tdir, f"profile_temporal_{rid}.json"), "w", encoding="utf-8") as f:
                    json.dump(prof, f, ensure_ascii=False, indent=2)

    else:
        best = model.suggest(
            last_nums=last_nums,
            n_samples=args.n_samples,
            top_k=args.top_k,
            seed=args.seed,
            w_recent=args.w_recent,
        )
        for rank, (score, nums, feats) in enumerate(best, start=1):
            d = {
                "bucket": "plain",
                "bucket_rank": rank,
                "nums": nums,
                "typical_raw": score,
                "strategy_score": score,
                "overlap_last": feats.get("overlap_last", 0),
                "feat_odds": feats.get("odds"),
                "feat_primes": feats.get("primes"),
                "feat_birthdays": feats.get("birthdays"),
                "feat_sum": feats.get("sum"),
                "feat_decades": None,
                "feat_maxrun": None,
                "feat_amp": None,
                "feat_digits": None,
                "hot_raw": None,
                "cold_raw": None,
                "temporal_raw": None,
                "z_typ": None,
                "z_hot": None,
                "z_cold": None,
            }
            predictions.append(_row_for_csv(rid, payload["ts"], d))

    print("\n" + "=" * 78)
    title = "PORTFOLIO MIXED-12 (4 estratégias x 3)" if args.portfolio == "mixed12" else f"TOP-{len(predictions)} (plain)"
    if args.portfolio == "mixed12" and args.temporal_plus:
        title += " + TEMPORAL PLUS-3"
    print(title)
    print("=" * 78)

    def bucket_title(b: str) -> str:
        return {
            "typical_top": "TÍPICOS (alto score típico)",
            "typical_diverse": "TÍPICOS (diversos entre si)",
            "hot_recency": "HOT RECENCY (quentes na janela recente)",
            "cold_overdue": "COLD/OVERDUE (mais tempo sem sair)",
            "temporal_period": "TEMPORAL (perfil do período-alvo)",
            "temporal_cold": "TEMPORAL + COLD (frio com viés temporal)",
            "temporal_typical": "TEMPORAL + TÍPICO (alto típico com viés temporal)",
            "plain": "PLAIN (top-k típico)",
        }.get(b, b)

    predictions_sorted = sorted(predictions, key=lambda r: (str(r["bucket"]), int(r["bucket_rank"])))
    current_bucket = None
    for row in predictions_sorted:
        if row["bucket"] != current_bucket:
            current_bucket = row["bucket"]
            print("\n" + bucket_title(str(current_bucket)))
            print("-" * 78)
        print(f"{int(row['bucket_rank']):>2}) {row['nums']} | sum={row.get('sum')} | odds={row.get('odds')} | primes={row.get('primes')} "
              f"| overlap_last={row.get('overlap_last')} | strategy={float(row.get('strategy_score') or 0.0):.3f}")

    meta_path = os.path.join(args.outdir, f"run_{rid}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.outdir, "predictions.csv")
    new_fields = list(predictions[0].keys()) if predictions else []
    union_fields = _safe_union_fieldnames(csv_path, new_fields)
    _rewrite_csv_with_union(csv_path, union_fields)

    if predictions:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=union_fields)
            if (not file_exists) or os.path.getsize(csv_path) == 0:
                w.writeheader()
            for r in predictions:
                out = {k: r.get(k, "") for k in union_fields}
                w.writerow(out)

    print(f"\nRun salvo: {meta_path}")
    print(f"Log consolidado: {csv_path}")

    if args.result:
        real = parse_nums(args.result)
        eval_rows = []
        for row in predictions:
            pred_nums = [int(x) for x in row["nums"].split()]
            hits = compute_hits(pred_nums, real)
            eval_rows.append({
                "run_id": row["run_id"],
                "bucket": row["bucket"],
                "bucket_rank": row["bucket_rank"],
                "pred": row["nums"],
                "real": " ".join(f"{n:02d}" for n in real),
                "hits": hits,
            })

        eval_path = os.path.join(args.outdir, f"eval_{rid}.csv")
        with open(eval_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            w.writeheader()
            for r in eval_rows:
                w.writerow(r)
        print(f"\nAvaliação salva: {eval_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
