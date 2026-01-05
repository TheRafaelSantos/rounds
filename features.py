# features.py
import math

PRIMOS = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59}

def count_odds(nums):
    return sum(n % 2 for n in nums)

def count_primes(nums):
    return sum(n in PRIMOS for n in nums)

def count_birthdays(nums):
    return sum(n <= 31 for n in nums)

def sum_nums(nums):
    return sum(nums)

def amplitude(nums):
    return max(nums) - min(nums)

def has_consecutive(nums):
    s = set(nums)
    return any((n+1) in s for n in nums)

def max_consecutive_run(nums):
    nums = sorted(nums)
    best = cur = 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best

def decade_buckets(nums):
    # [1-10],[11-20]...[51-60]
    buckets = [0]*6
    for n in nums:
        idx = min((n-1)//10, 5)
        buckets[idx] += 1
    return tuple(buckets)

def last_digit_profile(nums):
    # contagem por final 0..9
    cnt = [0]*10
    for n in nums:
        cnt[n % 10] += 1
    return tuple(cnt)

def overlap(a, b):
    return len(set(a) & set(b))

def score_likehood_log(p):
    # evita log(0)
    return math.log(max(p, 1e-12))
