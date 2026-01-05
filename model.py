# model.py
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data import DEZENAS
from features import (
    amplitude,
    count_birthdays,
    count_odds,
    count_primes,
    decade_buckets,
    distinct_last_digits,
    max_consecutive_run,
    overlap,
    score_likehood_log,
)
from analysis_temporal import build_temporal_profile, temporal_score

MAX_N = 60
P_IN_DRAW = 6 / 60  # 0.1

def _bin5(x: int) -> int:
    return int(round(int(x) / 5.0) * 5)

@dataclass
class Candidate:
    nums: List[int]
    typical_raw: float
    hot_raw: float
    cold_raw: float
    temporal_raw: Optional[float]
    feats: Dict[str, Any]

class ProfileModel:
    def __init__(self, df: pd.DataFrame, *, window_recent: int = 300):
        self.df = df.copy().sort_values("concurso").reset_index(drop=True)
        self.window_recent = int(max(1, window_recent))

        if "nums" not in self.df.columns:
            self.df["nums"] = self.df[DEZENAS].apply(lambda r: sorted(int(x) for x in r.tolist()), axis=1)

        self.existing = {tuple(x) for x in self.df["nums"].tolist()}
        self._n = len(self.df)

        self._last_seen_gap = self._compute_gaps()
        self._z_hot = self._compute_hot_z()

        self._dist_all = self._build_feature_dists(self.df)
        df_recent = self.df.iloc[max(0, self._n - self.window_recent):].copy()
        self._dist_recent = self._build_feature_dists(df_recent)

    def _compute_hot_z(self) -> np.ndarray:
        n_recent = min(self.window_recent, self._n)
        recent = self.df.iloc[self._n - n_recent:].copy()
        counts = np.zeros(MAX_N, dtype=float)
        for nums in recent["nums"].tolist():
            for n in nums:
                counts[int(n) - 1] += 1.0

        p_hat = counts / float(n_recent)
        exp = P_IN_DRAW
        sd = math.sqrt(exp * (1 - exp) / float(n_recent))
        if sd <= 0:
            return np.zeros(MAX_N, dtype=float)
        return (p_hat - exp) / sd

    def _compute_gaps(self) -> np.ndarray:
        last_idx = np.full(MAX_N, -1, dtype=int)
        for i, nums in enumerate(self.df["nums"].tolist()):
            for n in nums:
                last_idx[int(n) - 1] = i
        gap = (self._n - 1) - last_idx
        gap = np.where(last_idx < 0, self._n, gap)
        return gap.astype(float)

    def _featurize(self, nums: Sequence[int], last_nums: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        nums = sorted(int(x) for x in nums)
        f: Dict[str, Any] = {}
        f["odds"] = count_odds(nums)
        f["primes"] = count_primes(nums)
        f["birthdays"] = count_birthdays(nums)
        f["sum"] = int(sum(nums))
        f["sum_bin"] = _bin5(f["sum"])
        f["decades"] = decade_buckets(nums)
        f["maxrun"] = max_consecutive_run(nums)
        f["amp"] = amplitude(nums)
        f["amp_bin"] = _bin5(f["amp"])
        f["digits"] = distinct_last_digits(nums)
        f["overlap_last"] = overlap(nums, last_nums) if last_nums is not None else 0
        return f

    def _build_feature_dists(self, df: pd.DataFrame) -> Dict[str, Counter]:
        dists: Dict[str, Counter] = {
            "odds": Counter(),
            "primes": Counter(),
            "birthdays": Counter(),
            "sum_bin": Counter(),
            "decades": Counter(),
            "maxrun": Counter(),
            "amp_bin": Counter(),
            "digits": Counter(),
        }
        for nums in df["nums"].tolist():
            f = self._featurize(nums)
            for k in dists:
                dists[k][f[k]] += 1
        return dists

    def _p_from_dist(self, dist: Counter, key: Any, *, alpha: float = 1.0) -> float:
        total = sum(dist.values())
        k = len(dist) if len(dist) > 0 else 1
        return float((dist.get(key, 0) + alpha) / (total + alpha * k))

    def typical_score(self, nums: Sequence[int], *, w_recent: float = 0.55) -> float:
        f = self._featurize(nums)
        w = float(max(0.0, min(1.0, w_recent)))
        L_all = 0.0
        L_rec = 0.0

        weights = {
            "odds": 1.0,
            "primes": 0.8,
            "birthdays": 0.5,
            "sum_bin": 1.0,
            "decades": 0.8,
            "maxrun": 0.6,
            "amp_bin": 0.6,
            "digits": 0.4,
        }

        for k, wk in weights.items():
            p_all = self._p_from_dist(self._dist_all[k], f[k])
            p_rec = self._p_from_dist(self._dist_recent[k], f[k])
            L_all += wk * score_likehood_log(p_all)
            L_rec += wk * score_likehood_log(p_rec)

        return (1 - w) * L_all + w * L_rec

    def hot_score(self, nums: Sequence[int]) -> float:
        return float(np.mean([self._z_hot[int(n) - 1] for n in nums]))

    def cold_score(self, nums: Sequence[int]) -> float:
        exp_gap = 9.0
        gaps = [self._last_seen_gap[int(n) - 1] for n in nums]
        return float(np.mean([g / exp_gap for g in gaps]))

    def _gen_candidates(
        self,
        *,
        last_nums: Sequence[int],
        n_samples: int,
        seed: int,
        w_recent: float,
        max_overlap_last: int = 2,
        temporal_p: Optional[np.ndarray] = None,
    ) -> List[Candidate]:
        rnd = random.Random(int(seed))
        last_nums = sorted(int(x) for x in last_nums)

        out: List[Candidate] = []
        seen_local = set()

        for _ in range(int(n_samples)):
            nums = sorted(rnd.sample(range(1, MAX_N + 1), 6))
            t = tuple(nums)
            if t in self.existing or t in seen_local:
                continue
            seen_local.add(t)

            ov_last = overlap(nums, last_nums)
            if ov_last > int(max_overlap_last):
                continue

            typ = self.typical_score(nums, w_recent=w_recent)
            hot = self.hot_score(nums)
            cold = self.cold_score(nums)
            tmp = temporal_score(nums, temporal_p) if temporal_p is not None else None

            feats = self._featurize(nums, last_nums=last_nums)
            out.append(Candidate(nums=nums, typical_raw=typ, hot_raw=hot, cold_raw=cold, temporal_raw=tmp, feats=feats))

        return out

    def _pick_best(
        self,
        candidates: List[Candidate],
        *,
        objective_fn,
        k: int,
        already: List[List[int]],
        max_overlap_between_picks: int,
        extra_overlap_limit: Optional[int] = None,
    ) -> List[Candidate]:
        chosen: List[Candidate] = []
        ranked = sorted(candidates, key=objective_fn, reverse=True)

        for c in ranked:
            if len(chosen) >= k:
                break

            ok = True
            for p in already:
                if overlap(c.nums, p) > int(max_overlap_between_picks):
                    ok = False
                    break
            if not ok:
                continue

            if extra_overlap_limit is not None:
                for p in [x.nums for x in chosen]:
                    if overlap(c.nums, p) > int(extra_overlap_limit):
                        ok = False
                        break
            if not ok:
                continue

            chosen.append(c)
            already.append(c.nums)

        return chosen

    def _to_row(self, c: Candidate, *, bucket: str, rank: int, strategy_score: Optional[float] = None) -> Dict[str, Any]:
        f = c.feats
        return {
            "bucket": bucket,
            "bucket_rank": int(rank),
            "nums": c.nums,
            "typical_raw": float(c.typical_raw),
            "strategy_score": float(strategy_score if strategy_score is not None else c.typical_raw),
            "overlap_last": int(f.get("overlap_last", 0)),
            "feat_odds": int(f["odds"]),
            "feat_primes": int(f["primes"]),
            "feat_birthdays": int(f["birthdays"]),
            "feat_sum": int(f["sum"]),
            "feat_decades": f["decades"],
            "feat_maxrun": int(f["maxrun"]),
            "feat_amp": int(f["amp"]),
            "feat_digits": int(f["digits"]),
            "hot_raw": float(c.hot_raw),
            "cold_raw": float(c.cold_raw),
            "temporal_raw": (float(c.temporal_raw) if c.temporal_raw is not None else None),
            "z_typ": None,
            "z_hot": None,
            "z_cold": None,
        }

    def suggest_portfolio_mixed12(
        self,
        *,
        last_nums: Sequence[int],
        n_samples: int = 200_000,
        seed: int = 123,
        w_recent: float = 0.55,
        max_overlap_between_picks: int = 2,
    ) -> List[Dict[str, Any]]:
        last_nums = sorted(int(x) for x in last_nums)

        pool = self._gen_candidates(
            last_nums=last_nums,
            n_samples=n_samples,
            seed=seed,
            w_recent=w_recent,
            max_overlap_last=2,
            temporal_p=None,
        )

        already: List[List[int]] = []
        out: List[Dict[str, Any]] = []

        chosen = self._pick_best(pool, objective_fn=lambda c: c.typical_raw, k=3, already=already, max_overlap_between_picks=max_overlap_between_picks)
        out += [self._to_row(c, bucket="typical_top", rank=i+1) for i, c in enumerate(chosen)]

        chosen = self._pick_best(pool, objective_fn=lambda c: c.typical_raw, k=3, already=already, max_overlap_between_picks=max_overlap_between_picks, extra_overlap_limit=1)
        out += [self._to_row(c, bucket="typical_diverse", rank=i+1) for i, c in enumerate(chosen)]

        chosen = self._pick_best(pool, objective_fn=lambda c: (0.75 * c.hot_raw + 0.25 * c.typical_raw), k=3, already=already, max_overlap_between_picks=max_overlap_between_picks)
        out += [self._to_row(c, bucket="hot_recency", rank=i+1, strategy_score=(0.75 * c.hot_raw + 0.25 * c.typical_raw)) for i, c in enumerate(chosen)]

        chosen = self._pick_best(pool, objective_fn=lambda c: (0.75 * c.cold_raw + 0.25 * c.typical_raw), k=3, already=already, max_overlap_between_picks=max_overlap_between_picks)
        out += [self._to_row(c, bucket="cold_overdue", rank=i+1, strategy_score=(0.75 * c.cold_raw + 0.25 * c.typical_raw)) for i, c in enumerate(chosen)]

        return out

    def suggest_temporal_plus3(
        self,
        *,
        last_nums: Sequence[int],
        target_date: pd.Timestamp,
        n_samples: int = 200_000,
        seed: int = 123,
        w_recent: float = 0.55,
        max_overlap_between_picks: int = 2,
        already_picks: Optional[List[List[int]]] = None,
        recent_years_for_temporal: int = 0,
        next_is_special: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, object]]:
        prof = build_temporal_profile(self.df, pd.to_datetime(target_date), recent_years=recent_years_for_temporal)
        p_comb = np.array(prof["p_combined"], dtype=float)

        # Se o próximo concurso for especial: mistura com histórico só de especiais (Virada)
        special_prof = None
        if next_is_special and ("indicador_concurso_especial" in self.df.columns):
            df_sp = self.df[self.df["indicador_concurso_especial"].fillna(0).astype(int) == 2].copy()
            if len(df_sp) >= 10:
                special_prof = build_temporal_profile(df_sp, pd.to_datetime(target_date), recent_years=recent_years_for_temporal)
                p_sp = np.array(special_prof["p_combined"], dtype=float)
                w_gen, w_sp = 0.35, 0.65
                p_comb = np.exp((w_gen * np.log(np.maximum(p_comb, 1e-12)) + w_sp * np.log(np.maximum(p_sp, 1e-12))) / (w_gen + w_sp))

        last_nums = sorted(int(x) for x in last_nums)
        pool = self._gen_candidates(
            last_nums=last_nums,
            n_samples=n_samples,
            seed=seed,
            w_recent=w_recent,
            max_overlap_last=2,
            temporal_p=p_comb,
        )

        already = already_picks if already_picks is not None else []
        out: List[Dict[str, Any]] = []

        pick1 = self._pick_best(pool, objective_fn=lambda c: (1.00 * float(c.temporal_raw or -1e9) + 0.15 * c.typical_raw), k=1, already=already, max_overlap_between_picks=max_overlap_between_picks)
        if pick1:
            out.append(self._to_row(pick1[0], bucket="temporal_period", rank=1,
                                    strategy_score=(1.00 * float(pick1[0].temporal_raw or -1e9) + 0.15 * pick1[0].typical_raw)))

        pick2 = self._pick_best(pool, objective_fn=lambda c: (0.90 * float(c.temporal_raw or -1e9) + 0.60 * c.cold_raw + 0.10 * c.typical_raw), k=1, already=already, max_overlap_between_picks=max_overlap_between_picks)
        if pick2:
            out.append(self._to_row(pick2[0], bucket="temporal_cold", rank=1,
                                    strategy_score=(0.90 * float(pick2[0].temporal_raw or -1e9) + 0.60 * pick2[0].cold_raw + 0.10 * pick2[0].typical_raw)))

        pick3 = self._pick_best(pool, objective_fn=lambda c: (1.00 * c.typical_raw + 0.35 * float(c.temporal_raw or -1e9)), k=1, already=already, max_overlap_between_picks=max_overlap_between_picks)
        if pick3:
            out.append(self._to_row(pick3[0], bucket="temporal_typical", rank=1,
                                    strategy_score=(1.00 * pick3[0].typical_raw + 0.35 * float(pick3[0].temporal_raw or -1e9))))

        if special_prof is not None:
            prof["special_profile"] = special_prof

        return out, prof

    def suggest(
        self,
        *,
        last_nums: Sequence[int],
        n_samples: int = 200_000,
        top_k: int = 12,
        seed: int = 123,
        w_recent: float = 0.55,
    ) -> List[Tuple[float, List[int], Dict[str, Any]]]:
        pool = self._gen_candidates(last_nums=last_nums, n_samples=n_samples, seed=seed, w_recent=w_recent, max_overlap_last=2, temporal_p=None)
        ranked = sorted(pool, key=lambda c: c.typical_raw, reverse=True)[: int(top_k)]
        out = []
        for c in ranked:
            out.append((float(c.typical_raw), c.nums, c.feats))
        return out