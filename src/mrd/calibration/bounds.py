"""One-sided risk bounds (plan §7).

With zero errors, the one-sided 95% binomial upper bound is 1 - 0.05^(1/n): ~299 independent
negatives support a 1% bound, ~2,995 support 0.1%. Non-zero errors use the exact Clopper-Pearson
one-sided upper bound. `bonferroni_alpha` corrects for two constraints and the size of the
prespecified policy family.
"""
from __future__ import annotations

import math

from scipy.stats import beta


def clopper_pearson_upper(errors: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper (1-alpha) confidence bound on a binomial proportion."""
    if n <= 0:
        return float("nan")
    if errors >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, errors + 1, n - errors))


def zero_error_upper_bound(n: int, alpha: float = 0.05) -> float:
    return 1.0 - alpha ** (1.0 / n) if n > 0 else float("nan")


def negatives_needed(target_rate: float, alpha: float = 0.05) -> int:
    """Independent negatives needed so that zero errors certify `target_rate` at level alpha."""
    return math.ceil(math.log(alpha) / math.log(1.0 - target_rate))


def bonferroni_alpha(alpha: float, n_constraints: int, policy_family_size: int) -> float:
    return alpha / max(1, n_constraints * policy_family_size)


def independence_caveat(n_queries: int, n_clusters: int) -> str:
    """Text to accompany bounds: independent-query assumptions must be stated (plan §7)."""
    if n_clusters < n_queries:
        return (
            f"{n_queries} queries fall into {n_clusters} entity clusters; the binomial bound assumes independent "
            f"queries and should be read as optimistic. Effective n is at most {n_clusters}."
        )
    return f"{n_queries} queries in {n_clusters} distinct entity clusters; independence is plausible but unverified."
