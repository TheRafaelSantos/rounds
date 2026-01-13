# backtest.py
# Walk-forward backtest para Mega-Sena Lab (acadêmico) + (opcional) meta-aprendizado por bucket
#
# Exemplo:
#   python backtest.py --xlsx datalake_megasena.xlsx --n-eval 200 --n-samples 80000 --seed 123 --include-special --temporal-plus --megezord-plus
#
# Com meta-aprendizado:
#   python backtest.py --xlsx datalake_megasena.xlsx --n-eval 400 --min-history 300 --n-samples 150000 --seed 123 \
#       --include-special --temporal-plus --megezord-plus --learn-meta --meta-eta 0.25 --meta-forget 0.01 --meta-reward hits
#
# Saídas:
#   outputs/backtest_<runid>.csv
#   outputs/backtest_summary_<runid>.json
#   outputs/meta_weights_<runid>.json           (se --learn-meta)
#   outputs/meta_trace_<runid>.json             (se --learn-meta)
#   outputs/backtest_profiles/profiles_<runid>_<concurso>.json (a cada 50)

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from data import load_datalake_xlsx, DEZENAS
from model import ProfileModel

# opcional: meta-learner
try:
    from meta_learner import HedgeMetaLearner
except Exception:
    HedgeMetaLearner = None  # type: ignore

MAX_N = 60


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_nums(nums: Sequence[int]) -> List[int]:
    return sorted(int(x) for x in nums)


def compute_hits(pred: Sequence[int], real: Sequence[int]) -> int:
    return len(set(pred) & set(real))


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


def is_special_row(row: pd.Series) -> bool:
    if "indicador_concurso_especial" not in row.index:
        return False
    try:
        v = pd.to_numeric(row["indicador_concurso_especial"], errors="coerce")
        return int(0 if pd.isna(v) else v) == 2
    except Exception:
        return False


def _fmt_hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}"


def make_random_portfolio(
    *,
    k: int,
    rnd: random.Random,
    existing_set: set,
    last_nums: Sequence[int],
    max_overlap_last: int,
    max_overlap_between_picks: int,
    already_picks: Optional[List[List[int]]] = None,
) -> List[List[int]]:
    """
    Baseline random comparável:
    - evita repetir jogos já existentes no histórico (existing_set)
    - limita overlap com último resultado (max_overlap_last)
    - limita overlap entre picks do próprio portfolio (max_overlap_between_picks)
    """
    already = already_picks[:] if already_picks is not None else []
    out: List[List[int]] = []
    tries = 0

    while len(out) < k and tries < 300000:
        tries += 1
        nums = sorted(rnd.sample(range(1, MAX_N + 1), 6))
        t = tuple(nums)

        if t in existing_set:
            continue

        ov_last = len(set(nums) & set(last_nums))
        if ov_last > int(max_overlap_last):
            continue

        ok = True
        for p in already:
            if len(set(nums) & set(p)) > int(max_overlap_between_picks):
                ok = False
                break
        if not ok:
            continue

        out.append(nums)
        already.append(nums)

    return out


@dataclass
class PickRow:
    run_id: str
    ts: str
    eval_concurso: int
    eval_date: str
    last_concurso: int
    last_date: str

    bucket: str
    bucket_rank: int
    nums: str
    hits: int

    # scores
    strategy_score: Optional[float]
    typical_raw: Optional[float]
    hot_raw: Optional[float]
    cold_raw: Optional[float]
    temporal_raw: Optional[float]
    transition_raw: Optional[float]

    # z-scores (do model)
    z_typ: Optional[float]
    z_hot: Optional[float]
    z_cold: Optional[float]
    z_tr: Optional[float]

    # feats (para análise)
    overlap_last: Optional[int]
    feat_sum: Optional[int]
    feat_odds: Optional[int]
    feat_primes: Optional[int]
    feat_birthdays: Optional[int]
    feat_decades: Any
    feat_maxrun: Optional[int]
    feat_amp: Optional[int]
    feat_digits: Optional[int]


def dict_from_pickrow(r: PickRow) -> Dict[str, Any]:
    return {
        "run_id": r.run_id,
        "ts": r.ts,
        "eval_concurso": r.eval_concurso,
        "eval_date": r.eval_date,
        "last_concurso": r.last_concurso,
        "last_date": r.last_date,
        "bucket": r.bucket,
        "bucket_rank": r.bucket_rank,
        "nums": r.nums,
        "hits": r.hits,
        "strategy_score": r.strategy_score,
        "typical_raw": r.typical_raw,
        "hot_raw": r.hot_raw,
        "cold_raw": r.cold_raw,
        "temporal_raw": r.temporal_raw,
        "transition_raw": r.transition_raw,
        "z_typ": r.z_typ,
        "z_hot": r.z_hot,
        "z_cold": r.z_cold,
        "z_tr": r.z_tr,
        "overlap_last": r.overlap_last,
        "feat_sum": r.feat_sum,
        "feat_odds": r.feat_odds,
        "feat_primes": r.feat_primes,
        "feat_birthdays": r.feat_birthdays,
        "feat_decades": r.feat_decades,
        "feat_maxrun": r.feat_maxrun,
        "feat_amp": r.feat_amp,
        "feat_digits": r.feat_digits,
    }


def _to_datestr(x: Any) -> str:
    if hasattr(x, "date"):
        try:
            return str(x.date())
        except Exception:
            return str(x)
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest walk-forward do Mega-Sena Lab")
    ap.add_argument("--xlsx", default="datalake_megasena.xlsx")
    ap.add_argument("--outdir", default="outputs")

    ap.add_argument("--n-eval", type=int, default=200, help="Quantidade de concursos para avaliar (do final para trás).")
    ap.add_argument("--min-history", type=int, default=300, help="Histórico mínimo antes de começar a avaliar.")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--n-samples", type=int, default=80_000, help="n_samples do modelo durante backtest.")
    ap.add_argument("--w-recent", type=float, default=0.55)
    ap.add_argument("--window-recent", type=int, default=300)

    ap.add_argument("--max-overlap-picks", type=int, default=2, help="Overlap máximo entre picks do portfolio.")
    ap.add_argument("--max-overlap-last", type=int, default=2, help="Overlap máximo com o último resultado (baseline).")

    ap.add_argument("--include-special", action="store_true")
    ap.add_argument("--temporal-plus", action="store_true")
    ap.add_argument("--megezord-plus", action="store_true")
    ap.add_argument("--temporal-recent-years", type=int, default=0)

    # ---- Logging ----
    ap.add_argument("--log-every", type=int, default=10, help="Imprime progresso a cada N concursos (0=desliga).")

    # ---- Meta learning (fase 2) ----
    ap.add_argument("--learn-meta", action="store_true", help="Ativa meta-aprendizado de pesos por bucket.")
    ap.add_argument("--meta-eta", type=float, default=0.25, help="Taxa de aprendizado do meta (eta).")
    ap.add_argument("--meta-forget", type=float, default=0.01, help="Esquecimento 0..1 (0=desligado).")
    ap.add_argument(
        "--meta-reward",
        choices=["hits", "hit3"],
        default="hits",
        help="Reward por bucket: hits=max_hits_do_bucket, hit3=1 se max_hits>=3 senão 0.",
    )

    args = ap.parse_args()
    ensure_dir(args.outdir)

    if args.learn_meta and HedgeMetaLearner is None:
        raise RuntimeError("Você ativou --learn-meta, mas não consegui importar meta_learner.py (HedgeMetaLearner).")

    df = load_datalake_xlsx(args.xlsx).sort_values("concurso").reset_index(drop=True)
    df = ensure_nums_column(df)

    if "data_sorteio" not in df.columns:
        raise ValueError("Seu Excel precisa ter coluna data_sorteio para backtest.")

    df["data_sorteio"] = pd.to_datetime(df["data_sorteio"], errors="coerce")

    if (not args.include_special) and ("indicador_concurso_especial" in df.columns):
        df = df[df["indicador_concurso_especial"].fillna(0).astype(int) != 2].copy()
        df = df.sort_values("concurso").reset_index(drop=True)

    n_total = len(df)
    if n_total < (args.min_history + 10):
        raise ValueError(f"Poucos concursos ({n_total}) para backtest com min_history={args.min_history}.")

    n_eval = int(min(args.n_eval, max(0, n_total - args.min_history - 1)))
    if n_eval <= 0:
        raise ValueError("n_eval resultou em 0. Diminua --min-history ou aumente dataset.")

    end_idx = n_total - 1
    start_idx = end_idx - n_eval + 1

    now = dt.datetime.now().isoformat(timespec="seconds")
    payload = {
        "ts": now,
        "xlsx": os.path.basename(args.xlsx),
        "n_eval": int(n_eval),
        "min_history": int(args.min_history),
        "idx_range": [int(start_idx), int(end_idx)],
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "w_recent": float(args.w_recent),
        "window_recent": int(args.window_recent),
        "max_overlap_picks": int(args.max_overlap_picks),
        "max_overlap_last": int(args.max_overlap_last),
        "include_special": bool(args.include_special),
        "temporal_plus": bool(args.temporal_plus),
        "megezord_plus": bool(args.megezord_plus),
        "temporal_recent_years": int(args.temporal_recent_years),
        "learn_meta": bool(args.learn_meta),
        "meta_eta": float(args.meta_eta),
        "meta_forget": float(args.meta_forget),
        "meta_reward": str(args.meta_reward),
        "log_every": int(args.log_every),
    }
    rid = run_id_from_payload(payload)

    rows: List[PickRow] = []
    best_model_hits: List[int] = []
    best_baseline_hits: List[int] = []

    # ---- Meta state (persistente ao longo do backtest) ----
    meta = None
    meta_trace: List[Dict[str, Any]] = []
    if args.learn_meta:
        known = [
            "typical_top",
            "typical_diverse",
            "hot_recency",
            "cold_overdue",
            "temporal_period",
            "temporal_cold",
            "temporal_typical",
            "megezord_period",
            "megezord_typical",
            "megezord_cold",
        ]
        meta = HedgeMetaLearner.init(known, eta=args.meta_eta, forget=args.meta_forget)

    print(f"Backtest: avaliando {n_eval} concursos | idx {start_idx}..{end_idx} | run_id={rid}")

    # ---- Timers (para bloco/ETA) ----
    t0 = time.perf_counter()
    t_block = t0
    last_j_logged = 0
    width = max(4, len(str(n_eval)))

    for j, idx in enumerate(range(start_idx, end_idx + 1), start=1):
        train = df.iloc[:idx].copy()
        test = df.iloc[idx].copy()

        last_row = train.iloc[-1]
        last_concurso = int(last_row["concurso"])
        last_date = last_row["data_sorteio"]
        last_nums = normalize_nums(last_row["nums"])

        eval_concurso = int(test["concurso"])
        eval_date = test["data_sorteio"]
        real_nums = normalize_nums(test["nums"])

        next_is_special = bool(is_special_row(test))
        model = ProfileModel(train, window_recent=args.window_recent)

        # --- Portfolio base (mixed12) ---
        preds = model.suggest_portfolio_mixed12(
            last_nums=last_nums,
            n_samples=args.n_samples,
            seed=args.seed + idx,
            w_recent=args.w_recent,
            max_overlap_between_picks=args.max_overlap_picks,
        )
        all_pred_rows: List[Dict[str, Any]] = list(preds)

        # --- Temporal +3 ---
        temporal_prof = None
        if args.temporal_plus and hasattr(model, "suggest_temporal_plus3"):
            already_picks = [d["nums"] for d in all_pred_rows if "nums" in d]
            plus_rows, prof = model.suggest_temporal_plus3(
                last_nums=last_nums,
                target_date=pd.to_datetime(eval_date),
                n_samples=args.n_samples,
                seed=args.seed + idx + 7,
                w_recent=args.w_recent,
                max_overlap_between_picks=args.max_overlap_picks,
                already_picks=already_picks,
                recent_years_for_temporal=args.temporal_recent_years,
                next_is_special=next_is_special,
            )
            temporal_prof = prof
            all_pred_rows.extend(list(plus_rows))

        # --- Megezord +3 (temporal + transition) ---
        megezord_prof = None
        if args.megezord_plus and hasattr(model, "suggest_megezord_plus3"):
            already_picks = [d["nums"] for d in all_pred_rows if "nums" in d]
            mz_rows, mz_prof = model.suggest_megezord_plus3(
                last_nums=last_nums,
                target_date=pd.to_datetime(eval_date),
                n_samples=max(int(args.n_samples), 120_000),
                seed=args.seed + idx + 19,
                w_recent=args.w_recent,
                max_overlap_between_picks=args.max_overlap_picks,
                already_picks=already_picks,
                recent_years_for_temporal=args.temporal_recent_years,
                next_is_special=next_is_special,
            )
            megezord_prof = mz_prof
            all_pred_rows.extend(list(mz_rows))

        k_portfolio = len(all_pred_rows)

        # --- Baseline random com as mesmas restrições gerais ---
        rnd = random.Random(args.seed * 1000003 + idx)
        existing_set = {tuple(x) for x in train["nums"].tolist()}
        baseline_picks = make_random_portfolio(
            k=k_portfolio,
            rnd=rnd,
            existing_set=existing_set,
            last_nums=last_nums,
            max_overlap_last=args.max_overlap_last,
            max_overlap_between_picks=args.max_overlap_picks,
            already_picks=[],
        )

        # --- Registra picks do modelo ---
        model_hits_for_this_eval: List[int] = []
        for d in all_pred_rows:
            nums = normalize_nums(d["nums"])
            hits = compute_hits(nums, real_nums)
            model_hits_for_this_eval.append(hits)

            rows.append(
                PickRow(
                    run_id=rid,
                    ts=now,
                    eval_concurso=eval_concurso,
                    eval_date=_to_datestr(eval_date),
                    last_concurso=last_concurso,
                    last_date=_to_datestr(last_date),
                    bucket=str(d.get("bucket", "unknown")),
                    bucket_rank=int(d.get("bucket_rank", 1)),
                    nums=" ".join(f"{n:02d}" for n in nums),
                    hits=int(hits),
                    strategy_score=float(d["strategy_score"]) if d.get("strategy_score") is not None else None,
                    typical_raw=float(d["typical_raw"]) if d.get("typical_raw") is not None else None,
                    hot_raw=float(d["hot_raw"]) if d.get("hot_raw") is not None else None,
                    cold_raw=float(d["cold_raw"]) if d.get("cold_raw") is not None else None,
                    temporal_raw=float(d["temporal_raw"]) if d.get("temporal_raw") is not None else None,
                    transition_raw=float(d["transition_raw"]) if d.get("transition_raw") is not None else None,
                    z_typ=float(d["z_typ"]) if d.get("z_typ") is not None else None,
                    z_hot=float(d["z_hot"]) if d.get("z_hot") is not None else None,
                    z_cold=float(d["z_cold"]) if d.get("z_cold") is not None else None,
                    z_tr=float(d["z_tr"]) if d.get("z_tr") is not None else None,
                    overlap_last=int(d["overlap_last"]) if d.get("overlap_last") is not None else None,
                    feat_sum=int(d["feat_sum"]) if d.get("feat_sum") is not None else None,
                    feat_odds=int(d["feat_odds"]) if d.get("feat_odds") is not None else None,
                    feat_primes=int(d["feat_primes"]) if d.get("feat_primes") is not None else None,
                    feat_birthdays=int(d["feat_birthdays"]) if d.get("feat_birthdays") is not None else None,
                    feat_decades=d.get("feat_decades"),
                    feat_maxrun=int(d["feat_maxrun"]) if d.get("feat_maxrun") is not None else None,
                    feat_amp=int(d["feat_amp"]) if d.get("feat_amp") is not None else None,
                    feat_digits=int(d["feat_digits"]) if d.get("feat_digits") is not None else None,
                )
            )

        # --- Registra baseline ---
        baseline_hits_for_this_eval: List[int] = []
        for rnk, nums in enumerate(baseline_picks, start=1):
            hits = compute_hits(nums, real_nums)
            baseline_hits_for_this_eval.append(hits)

            rows.append(
                PickRow(
                    run_id=rid,
                    ts=now,
                    eval_concurso=eval_concurso,
                    eval_date=_to_datestr(eval_date),
                    last_concurso=last_concurso,
                    last_date=_to_datestr(last_date),
                    bucket="baseline_random",
                    bucket_rank=int(rnk),
                    nums=" ".join(f"{n:02d}" for n in nums),
                    hits=int(hits),
                    strategy_score=None,
                    typical_raw=None,
                    hot_raw=None,
                    cold_raw=None,
                    temporal_raw=None,
                    transition_raw=None,
                    z_typ=None,
                    z_hot=None,
                    z_cold=None,
                    z_tr=None,
                    overlap_last=None,
                    feat_sum=None,
                    feat_odds=None,
                    feat_primes=None,
                    feat_birthdays=None,
                    feat_decades=None,
                    feat_maxrun=None,
                    feat_amp=None,
                    feat_digits=None,
                )
            )

        # --- META update (fase 2): aprende pesos por bucket a cada concurso ---
        if meta is not None:
            bucket_to_hits: Dict[str, List[int]] = {}
            for d in all_pred_rows:
                b = str(d.get("bucket", "unknown"))
                nums = normalize_nums(d["nums"])
                h = compute_hits(nums, real_nums)
                bucket_to_hits.setdefault(b, []).append(int(h))

            rewards: Dict[str, float] = {}
            for b, hs in bucket_to_hits.items():
                best_h = max(hs) if hs else 0
                if args.meta_reward == "hit3":
                    rewards[b] = 1.0 if best_h >= 3 else 0.0
                else:
                    rewards[b] = float(best_h)

            meta.update(rewards)

            meta_trace.append(
                {
                    "eval_concurso": eval_concurso,
                    "eval_date": _to_datestr(eval_date),
                    "rewards": rewards,
                    "weights": dict(meta.weights),
                }
            )

        best_model_hits.append(int(max(model_hits_for_this_eval) if model_hits_for_this_eval else 0))
        best_baseline_hits.append(int(max(baseline_hits_for_this_eval) if baseline_hits_for_this_eval else 0))

        # ---- Log (bloco/total/ETA) ----
        do_log = False
        if int(args.log_every) > 0:
            if j == 1 or j == n_eval or (j % int(args.log_every) == 0):
                do_log = True

        if do_log:
            t_now = time.perf_counter()
            block_secs = t_now - t_block
            delta = j - last_j_logged
            if delta <= 0:
                delta = 1
            sec_per = block_secs / float(delta)

            total_secs = t_now - t0
            avg_secs = total_secs / float(max(1, j))
            eta_secs = avg_secs * float(max(0, n_eval - j))

            print(
                f"[{j:>{width}}/{n_eval}] concurso={eval_concurso} | best_model={best_model_hits[-1]} | best_rand={best_baseline_hits[-1]} | k={k_portfolio}"
                f" | bloco={_fmt_hms(block_secs)} ({sec_per:.3f}s/conc) | total={_fmt_hms(total_secs)} | ETA={_fmt_hms(eta_secs)}"
            )

            t_block = t_now
            last_j_logged = j

        # salva perfis de tempos em tempos (leve)
        if (temporal_prof is not None or megezord_prof is not None) and (j % 50 == 0 or j == n_eval):
            prof_dir = os.path.join(args.outdir, "backtest_profiles")
            ensure_dir(prof_dir)
            prof_path = os.path.join(prof_dir, f"profiles_{rid}_{eval_concurso}.json")
            with open(prof_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "eval_concurso": eval_concurso,
                        "eval_date": _to_datestr(eval_date),
                        "temporal_prof": temporal_prof,
                        "megezord_prof": megezord_prof,
                        "meta_weights": (dict(meta.weights) if meta is not None else None),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    # --- Export CSV detalhado ---
    out_csv = os.path.join(args.outdir, f"backtest_{rid}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        first = dict_from_pickrow(rows[0]) if rows else {}
        fieldnames = list(first.keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(dict_from_pickrow(r))

    # --- Resumo ---
    df_out = pd.DataFrame([dict_from_pickrow(r) for r in rows])

    def bucket_summary(dfx: pd.DataFrame) -> Dict[str, Any]:
        hits = dfx["hits"].astype(int)
        return {
            "n_picks": int(len(dfx)),
            "mean_hits": float(hits.mean()),
            "p_ge_3": float((hits >= 3).mean()),
            "p_ge_4": float((hits >= 4).mean()),
            "p_ge_5": float((hits >= 5).mean()),
            "max_hits": int(hits.max()) if len(hits) else 0,
        }

    by_bucket: Dict[str, Any] = {}
    for b, g in df_out.groupby("bucket"):
        by_bucket[str(b)] = bucket_summary(g)

    summary: Dict[str, Any] = {
        "run_id": rid,
        "payload": payload,
        "files": {"csv": out_csv},
        "best_of_portfolio": {
            "model_mean_best": float(np.mean(best_model_hits)) if best_model_hits else 0.0,
            "model_p_best_ge_3": float(np.mean([x >= 3 for x in best_model_hits])) if best_model_hits else 0.0,
            "model_p_best_ge_4": float(np.mean([x >= 4 for x in best_model_hits])) if best_model_hits else 0.0,
            "baseline_mean_best": float(np.mean(best_baseline_hits)) if best_baseline_hits else 0.0,
            "baseline_p_best_ge_3": float(np.mean([x >= 3 for x in best_baseline_hits])) if best_baseline_hits else 0.0,
            "baseline_p_best_ge_4": float(np.mean([x >= 4 for x in best_baseline_hits])) if best_baseline_hits else 0.0,
        },
        "by_bucket": by_bucket,
    }

    # salva meta weights + trace (fase 2)
    if meta is not None:
        meta_path = os.path.join(args.outdir, f"meta_weights_{rid}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": rid, "payload": payload, "weights": meta.weights}, f, ensure_ascii=False, indent=2)

        meta_trace_path = os.path.join(args.outdir, f"meta_trace_{rid}.json")
        with open(meta_trace_path, "w", encoding="utf-8") as f:
            json.dump(meta_trace, f, ensure_ascii=False, indent=2)

        summary["files"]["meta_weights"] = meta_path
        summary["files"]["meta_trace"] = meta_trace_path
        summary["meta_final_weights"] = dict(meta.weights)

    out_json = os.path.join(args.outdir, f"backtest_summary_{rid}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nOK.")
    print(f"CSV:  {out_csv}")
    print(f"JSON: {out_json}")

    if meta is not None:
        print(f"Meta weights: {summary['files']['meta_weights']}")
        print(f"Meta trace:   {summary['files']['meta_trace']}")

    print("\nResumo best-of-portfolio:")
    print(json.dumps(summary["best_of_portfolio"], ensure_ascii=False, indent=2))

    print("\nTop buckets por mean_hits:")
    items = sorted(by_bucket.items(), key=lambda kv: kv[1]["mean_hits"], reverse=True)
    for b, s in items[:12]:
        print(f"- {b}: mean={s['mean_hits']:.4f} | p>=3={s['p_ge_3']:.4f} | max={s['max_hits']} | n={s['n_picks']}")

    if meta is not None:
        print("\nMeta final weights (top 12):")
        topw = sorted(meta.weights.items(), key=lambda kv: kv[1], reverse=True)[:12]
        for b, wv in topw:
            print(f"- {b}: {wv:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())