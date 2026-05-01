"""Post-hoc probability calibration helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def fit_platt_scaler(logits: torch.Tensor, targets: torch.Tensor, min_samples: int = 25) -> tuple[float, float]:
    logits = logits.detach().float().reshape(-1)
    targets = targets.detach().float().reshape(-1)
    valid = torch.isfinite(logits) & torch.isfinite(targets)
    logits = logits[valid]
    targets = targets[valid]
    if logits.numel() < min_samples or targets.unique().numel() < 2:
        return 1.0, 0.0

    scale = torch.nn.Parameter(torch.tensor(1.0))
    bias = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.LBFGS([scale, bias], lr=0.25, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits((logits * scale) + bias, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scale.detach().cpu()), float(bias.detach().cpu())


def save_calibration_artifact(payload: dict, path: str | Path) -> None:
    Path(path).write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")


def load_calibration_artifact(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    return __import__("json").loads(artifact_path.read_text(encoding="utf-8"))


def apply_calibration(logits: torch.Tensor, artifact: dict | None) -> torch.Tensor:
    if artifact is None:
        return torch.sigmoid(logits)
    scales = torch.tensor(artifact["scales"], dtype=logits.dtype, device=logits.device)
    biases = torch.tensor(artifact["biases"], dtype=logits.dtype, device=logits.device)
    while scales.ndim < logits.ndim:
        scales = scales.unsqueeze(0)
        biases = biases.unsqueeze(0)
    return torch.sigmoid((logits * scales) + biases)


def calibration_error(y_true: torch.Tensor | list[float], y_prob: torch.Tensor | list[float], bins: int = 10) -> float:
    true = torch.as_tensor(y_true, dtype=torch.float32).reshape(-1)
    prob = torch.as_tensor(y_prob, dtype=torch.float32).reshape(-1)
    valid = torch.isfinite(true) & torch.isfinite(prob)
    true = true[valid]
    prob = prob[valid]
    if true.numel() == 0:
        return 0.0
    edges = torch.linspace(0.0, 1.0, steps=bins + 1)
    error = torch.tensor(0.0)
    n = float(true.numel())
    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not torch.any(mask):
            continue
        weight = float(mask.sum()) / n
        error = error + torch.abs(true[mask].mean() - prob[mask].mean()) * weight
    return float(error.item())
