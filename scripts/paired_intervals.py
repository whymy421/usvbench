"""Paired binary-outcome inference on one fixed-policy comparison (no numpy).

C7 STRATEGY section 6: keep the 2x2 table of (success_A, success_B), estimate
the paired risk difference with a paired score interval, test no-difference
with exact McNemar, never subtract two unpaired Wilson bounds, and never report
a [0, 0] interval when every pair agrees.

Interval: Tango (1998, Statistics in Medicine 17:891-908) asymptotic score
interval for Delta = p10 - p01 = P(A) - P(B). With b = #(A success, B fail),
c = #(A fail, B success) and n pairs,

    T(Delta) = (b - c - n*Delta) / sqrt(n * (2*q(Delta) + Delta*(1 - Delta)))

where q(Delta) is the restricted maximum-likelihood estimate of p01 under
p10 - p01 = Delta:

    q = (sqrt(W**2 + 8*n*c*Delta*(1 - Delta)) - W) / (4*n),
    W = -b - c + (2*n - b + c) * Delta.

The interval is {Delta : |T(Delta)| <= z}; T is monotone decreasing in Delta,
so both limits are found by bisection. At b = c = 0 it gives the nonzero
interval [-z^2/(n+z^2), z^2/(n+z^2)].

Assumes independent assigned scenario pairs. Clustered designs (reused layouts,
shared weather trajectories, several training seeds) need cluster-level
resampling instead; this function says nothing about them.
"""
from __future__ import annotations

import math
from statistics import NormalDist

METHOD = "tango-1998-paired-score"


def _counts(pairs):
    values = list(pairs)
    if not values:
        raise ValueError("paired interval requires a nonempty assigned population")
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2
           or any(type(v) is not bool for v in pair) for pair in values):
        raise ValueError("paired outcomes must be (bool, bool) pairs")
    n = len(values)
    b = sum(1 for a, c in values if a and not c)
    c = sum(1 for a, d in values if d and not a)
    both = sum(1 for a, d in values if a and d)
    return n, b, c, both


def tango_statistic(b, c, n, delta):
    w = -b - c + (2 * n - b + c) * delta
    root = w * w + 8.0 * n * c * delta * (1.0 - delta)
    q = (math.sqrt(max(root, 0.0)) - w) / (4.0 * n)
    variance = n * (2.0 * q + delta * (1.0 - delta))
    numerator = b - c - n * delta
    if variance <= 0.0:
        if numerator == 0:
            return 0.0
        return math.inf if numerator > 0 else -math.inf
    return numerator / math.sqrt(variance)


def tango_interval(b, c, n, confidence=0.95):
    for name, value in (("b", b), ("c", c), ("n", n)):
        if type(value) is not int or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    if n == 0 or b + c > n:
        raise ValueError("need n > 0 and b + c <= n")
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    estimate = (b - c) / n

    def solve(lo, hi, target):
        # T(lo) >= target >= T(hi); T decreasing in delta.
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if tango_statistic(b, c, n, mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    lower = -1.0 if tango_statistic(b, c, n, -1.0) <= z else solve(-1.0, estimate, z)
    upper = 1.0 if tango_statistic(b, c, n, 1.0) >= -z else solve(estimate, 1.0, -z)
    return max(-1.0, lower), min(1.0, upper)


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar p-value (binomial on the discordant pairs)."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1))
    return min(1.0, 2.0 * (tail / (1 << m)))


def paired_rate_interval(pairs, confidence=0.95):
    """A-minus-B success-rate difference for (success_A, success_B) pairs."""
    n, b, c, both = _counts(pairs)
    lo, hi = tango_interval(b, c, n, confidence)
    return {
        "n": n, "n11": both, "n10": b, "n01": c, "n00": n - both - b - c,
        "difference": (b - c) / n,
        "interval": [lo, hi],
        "confidence": confidence,
        "method": METHOD,
        "mcnemar_exact_p": mcnemar_exact_p(b, c),
    }


def wilson_interval(successes, n, confidence=0.95):
    """Wilson score interval for one fixed-policy success rate."""
    if type(successes) is not int or type(n) is not int or n <= 0 or not 0 <= successes <= n:
        raise ValueError("Wilson interval needs integers 0 <= successes <= n, n > 0")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / n
    scale = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return [max(0.0, center - half), min(1.0, center + half)]
