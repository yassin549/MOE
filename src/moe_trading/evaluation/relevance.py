"""Static feature/label relevance diagnostics for validation folds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
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


def label_permutation_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    permutations: int = 200,
    block_size: int = 128,
    seed: int = 7,
) -> tuple[PermutationResult, np.ndarray]:
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
    result = PermutationResult(p_value=p_value, effect_size=float(observed - null.mean()), observed=float(observed), null_mean=float(null.mean()))
    return result, null


def feature_permutation_test(
    y_true: np.ndarray,
    y_prob_full: np.ndarray,
    per_feature_probs: dict[str, list[np.ndarray]],
) -> dict[str, PermutationResult]:
    baseline = _auc(y_true, y_prob_full)
    out: dict[str, PermutationResult] = {}
    for feature, scores in per_feature_probs.items():
        values = np.asarray(scores, dtype=np.float64)
        null_mean = float(values.mean()) if values.size else baseline
        p_value = float((1.0 + (values <= baseline).sum()) / (1.0 + values.size)) if values.size else 1.0
        out[feature] = PermutationResult(
            p_value=p_value,
            effect_size=float(baseline - null_mean),
            observed=baseline,
            null_mean=null_mean,
        )
    return out


def summarize_fold_relevance(
    y_true: np.ndarray,
    y_prob_full: np.ndarray,
    y_prob_time_only: np.ndarray,
    per_feature_scores: dict[str, list[float]],
    setup_scores: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    manager_label_test, manager_null = label_permutation_test(y_true, y_prob_full)
    manager_full_auc = _auc(y_true, y_prob_full)
    manager_time_auc = _auc(y_true, y_prob_time_only)
    uplift = manager_full_auc - manager_time_auc
    setup_label_tests = {k: asdict(label_permutation_test(v[0], v[1])[0]) for k, v in setup_scores.items()}
    per_feature = feature_permutation_test(y_true, y_prob_full, {k: [float(x) for x in v] for k, v in per_feature_scores.items()})
    return {
        "manager_target": {
            "label_permutation": asdict(manager_label_test),
            "label_permutation_null": manager_null.tolist(),
            "full_auc": float(manager_full_auc),
            "time_only_auc": float(manager_time_auc),
            "full_feature_uplift": float(uplift),
        },
        "setup_targets": setup_label_tests,
        "feature_permutation": {k: asdict(v) for k, v in per_feature.items()},
    }


def aggregate_uplift_significance(fold_reports: list[dict[str, Any]], alpha: float = 0.05) -> dict[str, Any]:
    deltas = np.array([float(r["manager_target"]["full_feature_uplift"]) for r in fold_reports], dtype=np.float64)
    observed = float(deltas.mean()) if deltas.size else 0.0
    positive = int((deltas > 0).sum())
    n = int(deltas.size)
    p_value = float(1.0 if n == 0 else sum(comb(n, k) for k in range(positive, n + 1)) / (2**n))
    return {
        "num_folds": n,
        "mean_uplift": observed,
        "positive_folds": positive,
        "p_value": p_value,
        "significant": bool(p_value < alpha and observed > 0),
        "alpha": alpha,
    }
