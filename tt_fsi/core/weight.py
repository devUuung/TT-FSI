"""
Weight calculation for FSI correction term.

w(s, t) = (-1)^{ℓ-s} × (s/(ℓ+s)) × C(ℓ,s) × C(t-1,ℓ) / C(t+ℓ-1,ℓ+s)

Uses log-gamma for numerical stability with large binomial coefficients.
"""

import numpy as np
from scipy.special import gammaln


def log_binom(n: int, k: int) -> float:
    """
    Compute log(C(n,k)) using log-gamma for numerical stability.

    C(n,k) = n! / (k! (n-k)!)
    log C(n,k) = log(n!) - log(k!) - log((n-k)!)
               = gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
    """
    if k < 0 or k > n:
        return -np.inf  # log(0)
    if n < 0:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def weight_w(s: int, t: int, ell: int) -> float:
    """
    Compute the correction weight w(s, t, ℓ).

    w(s, t) = (-1)^{ℓ-s} × (s/(ℓ+s)) × C(ℓ,s) × C(t-1,ℓ) / C(t+ℓ-1,ℓ+s)

    Constraints:
        - s ≤ ℓ (truncation)
        - t > ℓ (correction only for |T| > ℓ)
        - s > 0 (must have at least one element in S)
        - t ≥ s (superset condition: T ⊇ S implies |T| ≥ |S|)

    Args:
        s: |S|, the size of the output subset (0 ≤ s ≤ ℓ)
        t: |T|, the size of the input subset (must have t > ℓ for correction)
        ell: interaction order bound ℓ

    Returns:
        Weight value w(s, t), or 0.0 if constraints not satisfied.
    """
    # Check constraints
    if s > ell:  # truncation: s ≤ ℓ
        return 0.0
    if t <= ell:  # correction only for |T| > ℓ
        return 0.0
    if s <= 0:  # s = 0 gives zero weight (s/(ℓ+s) = 0)
        return 0.0
    if t < s:  # superset condition violated
        return 0.0

    # Sign: (-1)^{ℓ - s}
    sign = (-1) ** (ell - s)

    # Scalar factor: s / (ℓ + s)
    scalar = s / (ell + s)

    # Log of binomial product:
    # log[C(ℓ,s) × C(t-1,ℓ) / C(t+ℓ-1,ℓ+s)]
    log_val = (log_binom(ell, s)
               + log_binom(t - 1, ell)
               - log_binom(t + ell - 1, ell + s))

    return sign * scalar * np.exp(log_val)


def weight_w_direct(s: int, t: int, ell: int) -> float:
    """
    Direct computation of weight (for verification, less stable for large values).

    Uses scipy.special.comb for comparison.
    """
    from scipy.special import comb

    if s > ell or t <= ell or s <= 0 or t < s:
        return 0.0

    sign = (-1) ** (ell - s)
    scalar = s / (ell + s)
    binom_val = comb(ell, s) * comb(t - 1, ell) / comb(t + ell - 1, ell + s)

    return sign * scalar * binom_val


def verify_weight_stability(max_d: int = 30, ell: int = 2) -> dict:
    """
    Verify numerical stability of weight_w across different (s, t) values.

    Compares log-stable version with direct computation.

    Returns:
        Dictionary with max absolute and relative errors.
    """
    max_abs_error = 0.0
    max_rel_error = 0.0
    error_cases = []

    for s in range(1, ell + 1):
        for t in range(ell + 1, max_d + 1):
            w_log = weight_w(s, t, ell)
            w_direct = weight_w_direct(s, t, ell)

            abs_error = abs(w_log - w_direct)
            rel_error = abs_error / (abs(w_direct) + 1e-15)

            if abs_error > max_abs_error:
                max_abs_error = abs_error
                error_cases.append({
                    's': s, 't': t, 'ell': ell,
                    'w_log': w_log, 'w_direct': w_direct,
                    'abs_error': abs_error, 'rel_error': rel_error
                })

            if rel_error > max_rel_error:
                max_rel_error = rel_error

    return {
        'max_abs_error': max_abs_error,
        'max_rel_error': max_rel_error,
        'worst_cases': error_cases[-5:] if error_cases else []
    }
