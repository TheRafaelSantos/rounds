# rounds.py
# Mega-Sena Lab (acadêmico) - Runner/CLI + (opcional) meta-weights do backtest
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from data import load_datalake_xlsx, DEZENAS
from model import ProfileModel


MAX_N = 60


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


def compute_overlap(a: Sequence[int], b: Sequence[int]) -> int:
    return len(set(a) & set(b))


def run_id_from_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:10]


def ensure_nums_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "nums" in df.columns:
        df["nums"] = df["nums"].apply(lambda x: normalize_nums(x))
        return df

    cols = [c for c in ["d1", "d2", "d3", "d4", "d5", "d6"] if c in df.columns]
    if len(cols) != 6:
        cols = [c for c in DEZENAS if c in df.columns]

    if len(cols) != 6:
        raise ValueError("Não consegui montar df['nums']: não achei d1..d6 nem DEZENAS completos no Excel.")

    df["nums"] = df[cols].apply(lambda r: sorted(int(x) for x in r.tolist()), axis=1)
    return df


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


def _find_latest_meta_weights(outdir: str) -> Optional[str]:
    patt = os.path.join(outdir, "meta_weights_*.json")
    files = glob.glob(patt)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _load_meta_weights(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # formato esperado do backtest:
    # {"run_id": "...", "payload": {...}, "weights": {"bucket": 0.123, ...}}
    if isinstance(obj, dict) and "weights" in obj and isinstance(obj["weights"], dict):
        w = obj["weights"]
    elif isinstance(obj, dict):
        # fallback: assume que o JSON já é o mapa bucket->peso
        w = obj
    else:
        raise ValueError("meta_weights JSON inválido.")

    out: Dict[str, float] = {}
    for k, v in w.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue

    if not out:
        raise ValueError("meta_weights vazio ou não numérico.")
    return out


def _meta_score(bucket: str, bucket_rank: Any, weights: Optional[Dict[str, float]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Retorna (meta_weight, meta_score).
    meta_score simples: weight / rank
    """
    if not weights:
        return None, None
    w = float(weights.get(bucket, 1.0))
    try:
        r = int(bucket_rank)
        r = max(1, r)
    except Exception:
        r = 999
    return w, (w / float(r))


def _row_for_csv(run_id: str, ts: str, d: Dict[str, Any], meta_weights: Optional[Dict[str, float]]) -> Dict[str, Any]:
    nums = d["nums"]
    nums_s = " ".join(f"{n:02d}" for n in nums)

    bucket = str(d.get("bucket"))
    bucket_rank = d.get("bucket_rank")
    mw, ms = _meta_score(bucket, bucket_rank, meta_weights)

    return {
        "run_id": run_id,
        "ts": ts,
        "bucket": bucket,
        "bucket_rank": bucket_rank,
        "nums": nums_s,
        "typical_raw": d.get("typical_raw"),
        "strategy_score": d.get("strategy_score"),
        "meta_weight": mw,
        "meta_score": ms,
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
        "transition_raw": d.get("transition_raw"),
        "z_typ": d.get("z_typ"),
        "z_hot": d.get("z_hot"),
        "z_cold": d.get("z_cold"),
        "z_tr": d.get("z_tr"),
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
    p.add_argument("--megezord-plus", action="store_true")
    p.add_argument("--target-date", default=None)
    p.add_argument("--temporal-recent-years", type=int, default=0)

    # ---- Fase 3: usar pesos meta aprendidos no backtest ----
    p.add_argument("--use-meta", action="store_true", help="Aplica meta-weights (aprendidos no backtest) para ranquear palpites.")
    p.add_argument("--meta-weights", default=None, help="Caminho para outputs/meta_weights_<runid>.json")
    p.add_argument("--meta-latest", action="store_true", help="Usa o arquivo meta_weights_*.json mais recente em --outdir")
    p.add_argument("--meta-top-k", type=int, default=6, help="Quantidade de palpites finais recomendados pelo META (greedy).")

    args = p.parse_args()
    ensure_dir(args.outdir)

    # ---- meta weights load ----
    meta_weights: Optional[Dict[str, float]] = None
    meta_weights_path: Optional[str] = None

    if args.use_meta:
        if args.meta_weights:
            meta_weights_path = args.meta_weights
        elif args.meta_latest:
            meta_weights_path = _find_latest_meta_weights(args.outdir)

        if meta_weights_path:
            try:
                meta_weights = _load_meta_weights(meta_weights_path)
                print(f"[META] weights carregados: {meta_weights_path} | buckets={len(meta_weights)}")
            except Exception as e:
                print(f"[META][AVISO] Falha ao carregar meta_weights ({meta_weights_path}): {e}")
                meta_weights = None
        else:
            print("[META][AVISO] --use-meta ligado, mas não achei --meta-weights nem --meta-latest encontrou arquivo.")
            meta_weights = None

    df = load_datalake_xlsx(args.xlsx).sort_values("concurso").reset_index(drop=True)
    df = ensure_nums_column(df)

    # Aviso se o último do Excel é especial e você não incluiu especiais
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
        "megezord_plus": bool(args.megezord_plus),
        "target_date": args.target_date,
        "temporal_recent_years": int(args.temporal_recent_years),
        "next_is_special": bool(args.next_is_special),
        "use_meta": bool(args.use_meta),
        "meta_weights_path": meta_weights_path,
        "meta_top_k": int(args.meta_top_k),
    }
    rid = run_id_from_payload(payload)

    predictions: List[Dict[str, Any]] = []

    # -------------------------
    # Portfolio base
    # -------------------------
    already_picks: List[List[int]] = []

    if args.portfolio == "mixed12":
        preds = model.suggest_portfolio_mixed12(
            last_nums=last_nums,
            n_samples=args.n_samples,
            seed=args.seed,
            w_recent=args.w_recent,
            max_overlap_between_picks=args.max_overlap_picks,
        )
        for d in preds:
            predictions.append(_row_for_csv(rid, payload["ts"], d, meta_weights))
        already_picks = [d["nums"] for d in preds]  # nums em lista[int]
    else:
        if not hasattr(model, "suggest"):
            raise RuntimeError("Seu ProfileModel não tem método suggest(). Use --portfolio mixed12.")
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
                "feat_decades": feats.get("decades"),
                "feat_maxrun": feats.get("maxrun"),
                "feat_amp": feats.get("amp"),
                "feat_digits": feats.get("digits"),
                "hot_raw": None,
                "cold_raw": None,
                "temporal_raw": None,
                "transition_raw": None,
                "z_typ": None,
                "z_hot": None,
                "z_cold": None,
                "z_tr": None,
            }
            predictions.append(_row_for_csv(rid, payload["ts"], d, meta_weights))

    # -------------------------
    # Target date (para temporal/megezord)
    # -------------------------
    target_date = None
    if args.target_date:
        target_date = pd.to_datetime(args.target_date)
    elif "data_proximo_concurso" in df.columns and (not df["data_proximo_concurso"].isna().all()):
        target_date = pd.to_datetime(df["data_proximo_concurso"].iloc[-1])

    # -------------------------
    # Temporal +3
    # -------------------------
    if args.temporal_plus:
        if target_date is None or pd.isna(target_date):
            print("\n[AVISO] temporal_plus ligado, mas não encontrei target_date.")
            print("        Passe --target-date YYYY-MM-DD ou garanta data_proximo_concurso no Excel.")
        else:
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
                predictions.append(_row_for_csv(rid, payload["ts"], d, meta_weights))
                already_picks.append(d["nums"])

            tdir = os.path.join(args.outdir, "temporal")
            ensure_dir(tdir)
            with open(os.path.join(tdir, f"profile_temporal_{rid}.json"), "w", encoding="utf-8") as f:
                json.dump(prof, f, ensure_ascii=False, indent=2)

    # -------------------------
    # MEGEZORD +3
    # -------------------------
    if args.megezord_plus:
        if target_date is None or pd.isna(target_date):
            print("\n[AVISO] megezord_plus ligado, mas não encontrei target_date.")
            print("        Passe --target-date YYYY-MM-DD ou garanta data_proximo_concurso no Excel.")
        else:
            mega_rows, mega_prof = model.suggest_megezord_plus3(
                last_nums=last_nums,
                target_date=target_date,
                n_samples=max(args.n_samples, 250_000),
                seed=args.seed + 777,
                w_recent=args.w_recent,
                max_overlap_between_picks=args.max_overlap_picks,
                already_picks=already_picks,
                recent_years_for_temporal=args.temporal_recent_years,
                next_is_special=args.next_is_special,
            )
            for d in mega_rows:
                predictions.append(_row_for_csv(rid, payload["ts"], d, meta_weights))
                already_picks.append(d["nums"])

            mdir = os.path.join(args.outdir, "megezord")
            ensure_dir(mdir)
            with open(os.path.join(mdir, f"profile_megezord_{rid}.json"), "w", encoding="utf-8") as f:
                json.dump(mega_prof, f, ensure_ascii=False, indent=2)

    # -------------------------
    # Print portfolio por bucket
    # -------------------------
    print("\n" + "=" * 78)
    title = "PORTFOLIO MIXED-12 (4 estratégias x 3)" if args.portfolio == "mixed12" else f"TOP-{len(predictions)} (plain)"
    if args.temporal_plus:
        title += " + TEMPORAL PLUS-3"
    if args.megezord_plus:
        title += " + MEGEZORD PLUS-3"
    if args.use_meta and meta_weights:
        title += " + META-RANK"
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
            "temporal_typical": "TEMPORAL + TÍPICO (típico com viés temporal)",
            "megezord_period": "MEGEZORD (TEMPORAL + TRANSIÇÃO p1..p6)",
            "megezord_typical": "MEGEZORD (TEMPORAL + TRANSIÇÃO + TÍPICO)",
            "megezord_cold": "MEGEZORD (TEMPORAL + TRANSIÇÃO + COLD)",
            "plain": "PLAIN (top-k típico)",
        }.get(b, b)

    predictions_sorted = sorted(predictions, key=lambda r: (str(r["bucket"]), int(r["bucket_rank"] or 999)))
    current_bucket = None
    for row in predictions_sorted:
        if row["bucket"] != current_bucket:
            current_bucket = row["bucket"]
            print("\n" + bucket_title(str(current_bucket)))
            print("-" * 78)
        meta_part = ""
        if args.use_meta and meta_weights:
            meta_part = f" | meta_w={float(row.get('meta_weight') or 0.0):.6f} meta_s={float(row.get('meta_score') or 0.0):.6f}"
        print(
            f"{int(row['bucket_rank'] or 0):>2}) {row['nums']} | sum={row.get('sum')} | odds={row.get('odds')} | primes={row.get('primes')} "
            f"| overlap_last={row.get('overlap_last')} | strategy={float(row.get('strategy_score') or 0.0):.3f}{meta_part}"
        )

    # -------------------------
    # META PICKS (fase 3): top-k por meta_score + restrição de overlap entre os escolhidos
    # -------------------------
    if args.use_meta and meta_weights:
        ranked = sorted(predictions, key=lambda r: float(r.get("meta_score") or 0.0), reverse=True)
        chosen: List[Dict[str, Any]] = []
        chosen_nums: List[List[int]] = []

        for row in ranked:
            if len(chosen) >= int(args.meta_top_k):
                break
            nums = [int(x) for x in row["nums"].split()]
            ok = True
            for prev in chosen_nums:
                if compute_overlap(nums, prev) > int(args.max_overlap_picks):
                    ok = False
                    break
            if not ok:
                continue
            chosen.append(row)
            chosen_nums.append(nums)

        print("\n" + "=" * 78)
        print(f"META PICKS (TOP-{len(chosen)}): selecionados por meta_score (weight/rank) + overlap<= {args.max_overlap_picks}")
        print("=" * 78)
        for i, row in enumerate(chosen, start=1):
            print(
                f"{i:>2}) {row['nums']} | bucket={row['bucket']}#{row['bucket_rank']} "
                f"| meta_w={float(row.get('meta_weight') or 0.0):.6f} meta_s={float(row.get('meta_score') or 0.0):.6f}"
            )

    # -------------------------
    # Save meta + consolidated CSV
    # -------------------------
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

    # -------------------------
    # Evaluate if result passed
    # -------------------------
    if args.result:
        real = parse_nums(args.result)
        eval_rows = []
        for row in predictions:
            pred_nums = [int(x) for x in row["nums"].split()]
            hits = len(set(pred_nums) & set(real))
            eval_rows.append(
                {
                    "run_id": row["run_id"],
                    "bucket": row["bucket"],
                    "bucket_rank": row["bucket_rank"],
                    "pred": row["nums"],
                    "real": " ".join(f"{n:02d}" for n in real),
                    "hits": hits,
                }
            )

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
