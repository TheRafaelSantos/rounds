# calibrate_temporal_weights.py
# Calibra pesos do TEMPORAL PLUS-3 via backtest time-split (sem vazamento)
#
# Exemplo:
#   python calibrate_temporal_weights.py --xlsx datalake_megasena.xlsx --n-eval 400 --n-rand 4000 --seed 123
#
import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data import load_datalake_xlsx, DEZENAS
from model import ProfileModel
from analysis_temporal import build_temporal_profile, temporal_score


MAX_N = 60


@dataclass
class RowScores:
    typ: float
    cold: float
    hot: float
    tmp: float


def _z(x: np.ndarray) -> np.ndarray:
    m = float(np.mean(x))
    s = float(np.std(x))
    if s <= 1e-12:
        return (x - m) * 0.0
    return (x - m) / s


def _percentile_of_target(scores: np.ndarray, target_idx: int) -> float:
    # percentil = % de candidatos com score menor que o do target
    t = float(scores[target_idx])
    return float(np.mean(scores < t))


def _ensure_nums(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "nums" not in df.columns:
        df["nums"] = df[DEZENAS].apply(lambda r: sorted(int(x) for x in r.tolist()), axis=1)
    return df


def _sample_unique_tickets(rng: random.Random, n_rand: int, forbid: Optional[set] = None) -> List[Tuple[int, ...]]:
    out = []
    seen = set() if forbid is None else set(forbid)
    while len(out) < n_rand:
        nums = tuple(sorted(rng.sample(range(1, MAX_N + 1), 6)))
        if nums in seen:
            continue
        seen.add(nums)
        out.append(nums)
    return out


def eval_weights(
    df_all: pd.DataFrame,
    *,
    window_recent: int,
    n_eval: int,
    n_rand: int,
    burn_in: int,
    seed: int,
    recent_years_for_temporal: int,
    include_special: bool,
    only_special: bool,
    k_period_typ: float,
    k_cold: float,
    k_cold_typ: float,
    k_typ_temp: float,
) -> Dict[str, float]:
    df = df_all.copy()

    if (not include_special) and ("indicador_concurso_especial" in df.columns):
        df = df[df["indicador_concurso_especial"].fillna(1).astype(int) != 2].copy()

    if only_special and ("indicador_concurso_especial" in df.columns):
        df = df[df["indicador_concurso_especial"].fillna(0).astype(int) == 2].copy()

    df = _ensure_nums(df).sort_values("concurso").reset_index(drop=True)
    df = df.dropna(subset=["data_sorteio"]).copy()
    df["data_sorteio"] = pd.to_datetime(df["data_sorteio"])

    n = len(df)
    if n < burn_in + 2:
        raise ValueError(f"Poucos concursos para backtest: n={n}, burn_in={burn_in}")

    # vamos avaliar os últimos n_eval pontos, mas nunca além do que existe
    end = n - 2  # precisamos de t+1
    start = max(burn_in, end - int(n_eval) + 1)

    rng = random.Random(int(seed))

    per_period = []
    per_cold = []
    per_typ = []

    for t in range(start, end + 1):
        train = df.iloc[: t + 1].copy()
        target = df.iloc[t + 1]

        # modelo treinado só no passado
        model = ProfileModel(train, window_recent=window_recent)

        target_date = pd.to_datetime(target["data_sorteio"])
        prof = build_temporal_profile(train, target_date, recent_years=recent_years_for_temporal)
        p_comb = np.array(prof["p_combined"], dtype=float)

        # pool: random + target
        target_ticket = tuple(int(x) for x in target["nums"])
        rand_tickets = _sample_unique_tickets(rng, int(n_rand), forbid={target_ticket})
        tickets = rand_tickets + [target_ticket]
        target_idx = len(tickets) - 1

        # scores raw
        rows = []
        for nums in tickets:
            nums_list = list(nums)
            rows.append(RowScores(
                typ=float(model.typical_score(nums_list, w_recent=0.55)),
                cold=float(model.cold_score(nums_list)),
                hot=float(model.hot_score(nums_list)),
                tmp=float(temporal_score(nums_list, p_comb)),
            ))

        typ = np.array([r.typ for r in rows], dtype=float)
        cold = np.array([r.cold for r in rows], dtype=float)
        hot = np.array([r.hot for r in rows], dtype=float)
        tmp = np.array([r.tmp for r in rows], dtype=float)

        # z-score por rodada (recomendado)
        z_typ = _z(typ)
        z_cold = _z(cold)
        z_hot = _z(hot)
        z_tmp = _z(tmp)

        # 3 objetivos do TEMPORAL PLUS-3 (em z-score)
        score_period = (1.0 * z_tmp + float(k_period_typ) * z_typ)
        score_cold   = (1.0 * z_tmp + float(k_cold) * z_cold + float(k_cold_typ) * z_typ)
        score_typ    = (1.0 * z_typ + float(k_typ_temp) * z_tmp)

        per_period.append(_percentile_of_target(score_period, target_idx))
        per_cold.append(_percentile_of_target(score_cold, target_idx))
        per_typ.append(_percentile_of_target(score_typ, target_idx))

    return {
        "n_points": float(len(per_period)),
        "mean_pct_period": float(np.mean(per_period)),
        "mean_pct_cold": float(np.mean(per_cold)),
        "mean_pct_typ": float(np.mean(per_typ)),
        "mean_pct_avg3": float(np.mean([(a+b+c)/3.0 for a,b,c in zip(per_period, per_cold, per_typ)])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--window-recent", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--n-rand", type=int, default=4000)
    ap.add_argument("--burn-in", type=int, default=600)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--recent-years-temporal", type=int, default=0)
    ap.add_argument("--include-special", action="store_true")
    ap.add_argument("--only-special", action="store_true")
    ap.add_argument("--trials", type=int, default=120)

    # ranges dos pesos (z-score -> ranges estáveis)
    ap.add_argument("--k-period-typ-max", type=float, default=1.0)
    ap.add_argument("--k-cold-max", type=float, default=1.5)
    ap.add_argument("--k-cold-typ-max", type=float, default=0.6)
    ap.add_argument("--k-typ-temp-max", type=float, default=1.0)

    ap.add_argument("--out", default="outputs/calibration_temporal.json")
    args = ap.parse_args()

    df = load_datalake_xlsx(args.xlsx)
    df = _ensure_nums(df)

    rng = random.Random(int(args.seed))

    best = None
    best_params = None
    best_detail = None

    for i in range(int(args.trials)):
        # random search simples
        k_period_typ = rng.random() * float(args.k_period_typ_max)
        k_cold = rng.random() * float(args.k_cold_max)
        k_cold_typ = rng.random() * float(args.k_cold_typ_max)
        k_typ_temp = rng.random() * float(args.k_typ_temp_max)

        detail = eval_weights(
            df,
            window_recent=int(args.window_recent),
            n_eval=int(args.n_eval),
            n_rand=int(args.n_rand),
            burn_in=int(args.burn_in),
            seed=int(args.seed) + 17 * i,  # muda o pool por trial
            recent_years_for_temporal=int(args.recent_years_temporal),
            include_special=bool(args.include_special),
            only_special=bool(args.only_special),
            k_period_typ=k_period_typ,
            k_cold=k_cold,
            k_cold_typ=k_cold_typ,
            k_typ_temp=k_typ_temp,
        )

        score = detail["mean_pct_avg3"]
        if (best is None) or (score > best):
            best = score
            best_params = {
                "k_period_typ": float(k_period_typ),
                "k_cold": float(k_cold),
                "k_cold_typ": float(k_cold_typ),
                "k_typ_temp": float(k_typ_temp),
            }
            best_detail = detail

        print(f"[{i+1:03d}/{int(args.trials)}] avg3={detail['mean_pct_avg3']:.4f} "
              f"(period={detail['mean_pct_period']:.4f}, cold={detail['mean_pct_cold']:.4f}, typ={detail['mean_pct_typ']:.4f}) "
              f"| best={best:.4f}")

    out = {
        "best_score_mean_pct_avg3": float(best),
        "best_params": best_params,
        "best_detail": best_detail,
        "notes": {
            "metric": "percentil médio do concurso real vs tickets aleatórios (quanto mais alto, melhor). Em loteria justa tende a ~0.50.",
            "zscore": "pesos calibrados em z-score por rodada (recomendado).",
        },
        "args": vars(args),
    }

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== MELHOR ===")
    print(json.dumps(out["best_params"], indent=2))
    print(f"avg3={out['best_score_mean_pct_avg3']:.4f}")
    print(f"Salvo em: {args.out}")


if __name__ == "__main__":
    main()