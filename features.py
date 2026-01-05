# features.py
from __future__ import annotations

import math
from typing import Sequence, Tuple

PRIMOS = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59}

def count_odds(nums: Sequence[int]) -> int:
    return sum(n % 2 for n in nums)

def count_primes(nums: Sequence[int]) -> int:
    return sum(n in PRIMOS for n in nums)

def count_birthdays(nums: Sequence[int]) -> int:
    return sum(n <= 31 for n in nums)

def amplitude(nums: Sequence[int]) -> int:
    nums = list(nums)
    return int(max(nums) - min(nums))

def decade_buckets(nums: Sequence[int]) -> Tuple[int, int, int, int, int, int]:
    """
    Retorna contagem em 6 faixas:
      01-10, 11-20, 21-30, 31-40, 41-50, 51-60
    """
    bins = [0, 0, 0, 0, 0, 0]
    for n in nums:
        idx = (int(n) - 1) // 10  # 0..5
        idx = max(0, min(5, idx))
        bins[idx] += 1
    return tuple(bins)  # type: ignore[return-value]

def distinct_last_digits(nums: Sequence[int]) -> int:
    return len({int(n) % 10 for n in nums})

def max_consecutive_run(nums: Sequence[int]) -> int:
    nums = sorted(int(n) for n in nums)
    best = cur = 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best

def overlap(a: Sequence[int], b: Sequence[int]) -> int:
    return len(set(a) & set(b))

def score_likehood_log(p: float) -> float:
    return math.log(max(float(p), 1e-12))