"""Static feature/label relevance diagnostics for validation folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PermutationResult:
    p_value: float
    effect_size: float
    observed: float
    null_mean: float


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _block_indices(size: int, block_size: int) -> list[np.ndarray]:
    return [np.arange(i, min(i + block_size, size)) for i in range(0, size, block_size)]


def label_permutation_test(y_true: np.ndarray, y_prob: np.ndarray, permutations: int = 200, block_size: int = 128, seed: int = 7) -> PermutationResult:
    rng = np.random.default_rng(seed)
    observed = _auc(y_true, y_prob)
    blocks = _block_indices(len(y_true), max(1, block_size))
    null = np.zeros(permutations, dtype=np.float64)
    for i in range(permutations):
        shuffled = y_true.copy()
        for block in blocks:
            shuffled[block] = shuffled[block][rng.permutation(len(block))]
        null[i] = _auc(shuffled, y_prob)
    p_value = float((1.0 + (null >= observed).sum()) / (1.0 + permutations))
    effect = float(observed - null.mean())
    return PermutationResult(p_value=p_value, effect_size=effect, observed=float(observed), null_mean=float(null.mean()))


def feature_permutation_test(y_true: np.ndarray, y_prob_full: np.ndarray, per_feature_probs: dict[str, np.ndarray]) -> dict[str, PermutationResult]:
    baseline = _auc(y_true, y_prob_full)
    out: dict[str, PermutationResult] = {}
    for feature, probs in per_feature_probs.items():
        perm_score = _auc(y_true, probs)
        effect = baseline - perm_score
        out[feature] = PermutationResult(
            p_value=float(1.0 if effect <= 0 else 0.0),
            effect_size=float(effect),
            observed=float(perm_score),
            null_mean=float(baseline),
        )
    return out


def summarize_fold_relevance(
    y_true: np.ndarray,
    y_prob_full: np.ndarray,
    y_prob_time_only: np.ndarray,
    per_feature_probs: dict[str, np.ndarray],
    setup_scores: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    manager_label_test = label_permutation_test(y_true, y_prob_full)
    manager_full_auc = _auc(y_true, y_prob_full)
    manager_time_auc = _auc(y_true, y_prob_time_only)
    uplift = manager_full_auc - manager_time_auc
    setup_label_tests = {k: label_permutation_test(v[0], v[1]).__dict__ for k, v in setup_scores.items()}
    per_feature = feature_permutation_test(y_true, y_prob_full, per_feature_probs)
    return {
        "manager_target": {
            "label_permutation": manager_label_test.__dict__,
            "full_auc": float(manager_full_auc),
            "time_only_auc": float(manager_time_auc),
            "full_feature_uplift": float(uplift),
        },
        "setup_targets": setup_label_tests,
        "feature_permutation": {k: v.__dict__ for k, v in per_feature.items()},
    }


def aggregate_uplift_significance(fold_reports: list[dict[str, Any]], alpha: float = 0.05) -> dict[str, Any]:
    uplifts = np.array([float(r["manager_target"]["full_feature_uplift"]) for r in fold_reports], dtype=np.float64)
    null = np.array([float(r["manager_target"]["label_permutation"]["null_mean"]) for r in fold_reports], dtype=np.float64)
    deltas = uplifts - null
    observed = float(deltas.mean()) if deltas.size else 0.0
    # Sign test over contiguous folds
    positive = int((deltas > 0).sum())
    n = int(deltas.size)
    p_value = float(1.0 if n == 0 else sum(np.math.comb(n, k) for k in range(positive, n + 1)) / (2**n))
    return {
        "num_folds": n,
        "mean_uplift_minus_null": observed,
        "positive_folds": positive,
        "p_value": p_value,
        "significant": bool(p_value < alpha and observed > 0),
        "alpha": alpha,
    }
