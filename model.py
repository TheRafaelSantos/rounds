# model.py
import random
from collections import Counter
from features import (
    count_odds, count_primes, count_birthdays, sum_nums, decade_buckets,
    max_consecutive_run, overlap, score_likehood_log
)

MAX_N = 60
PICK = 6

class ProfileModel:
    def __init__(self, df, window_recent=300):
        self.df = df
        self.window_recent = window_recent

        # histórico inteiro
        self.dist_all = self._build_distributions(df)

        # recorte recente
        recent = df.tail(window_recent) if len(df) >= window_recent else df
        self.dist_recent = self._build_distributions(recent)

    def _build_distributions(self, df):
        odds = Counter()
        primes = Counter()
        birthdays = Counter()
        sums = Counter()
        decades = Counter()
        maxrun = Counter()

        for nums in df["nums"]:
            odds[count_odds(nums)] += 1
            primes[count_primes(nums)] += 1
            birthdays[count_birthdays(nums)] += 1
            sums[sum_nums(nums)] += 1
            decades[decade_buckets(nums)] += 1
            maxrun[max_consecutive_run(nums)] += 1

        n = len(df)
        return {
            "n": n,
            "odds": odds,
            "primes": primes,
            "birthdays": birthdays,
            "sums": sums,
            "decades": decades,
            "maxrun": maxrun,
        }

    def _p(self, dist, key, value):
        # Laplace smoothing simples (evita zero)
        c = dist[key][value]
        n = dist["n"]
        return (c + 1) / (n + len(dist[key]) + 1)

    def score(self, cand_nums, last_nums=None, w_recent=0.55):
        # mistura: recente + histórico
        def mix_prob(key, val):
            p_r = self._p(self.dist_recent, key, val)
            p_a = self._p(self.dist_all, key, val)
            return w_recent * p_r + (1 - w_recent) * p_a

        o = count_odds(cand_nums)
        p = count_primes(cand_nums)
        b = count_birthdays(cand_nums)
        s = sum_nums(cand_nums)
        d = decade_buckets(cand_nums)
        r = max_consecutive_run(cand_nums)

        L = 0.0
        L += 1.00 * score_likehood_log(mix_prob("odds", o))
        L += 0.60 * score_likehood_log(mix_prob("primes", p))
        L += 0.35 * score_likehood_log(mix_prob("birthdays", b))
        L += 0.25 * score_likehood_log(mix_prob("sums", s))
        L += 0.40 * score_likehood_log(mix_prob("decades", d))
        L += 0.25 * score_likehood_log(mix_prob("maxrun", r))

        # penalidade por overlap com o último concurso (se quiser)
        if last_nums is not None:
            ov = overlap(cand_nums, last_nums)
            if ov >= 3:
                L -= 2.0  # raríssimo e geralmente “feio” no teu insight
            elif ov == 2:
                L -= 0.6

        return L, {"odds": o, "primes": p, "birthdays": b, "sum": s, "decades": d, "maxrun": r}

    def suggest(self, last_nums, n_samples=200_000, top_k=20, seed=123):
        random.seed(seed)
        seen = set(tuple(x) for x in self.df["nums"])
        best = []

        for _ in range(n_samples):
            cand = sorted(random.sample(range(1, MAX_N+1), PICK))
            t = tuple(cand)
            if t in seen:
                continue

            sc, feats = self.score(cand, last_nums=last_nums)

            best.append((sc, cand, feats))
        best.sort(key=lambda x: x[0], reverse=True)
        return best[:top_k]
