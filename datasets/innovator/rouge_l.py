#!/usr/bin/env python3
"""Token-level ROUGE-L via numba-JIT LCS DP."""

import numpy as np
from numba import jit


@jit(nopython=True)
def _compute_dp(ref, pred):
    m, n = len(ref), len(pred)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp


def rouge_l_f1(ref_tokens: np.ndarray, pred_tokens: np.ndarray, max_len: int = 2048) -> float:
    ref = ref_tokens[:max_len]
    pred = pred_tokens[:max_len]
    if len(ref) == 0 or len(pred) == 0:
        return 0.0
    dp = _compute_dp(ref, pred)
    lcs = dp[len(ref)][len(pred)]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0
