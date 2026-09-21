"""Decomposable scoring control; changes rewards, never backbone attention."""

import torch

from .base import CompositeReward


class IndependentCellReward(CompositeReward):
    """Average individual-cell versions of the same three reward components.

    Every singleton sees the same control reference and training priors. Raw
    singleton scores are averaged BEFORE particle-wise normalization. This is
    a deliberately decomposable comparator, not the original population MMD.
    """

    def compute(self, x_pred, condition, timestep, **kwargs):
        n, m, g = x_pred.shape
        scores = []
        for reward in self.rewards:
            # Chunk singleton MMD to bound memory with large reference sets.
            flat = x_pred.reshape(n * m, 1, g)
            raw = (
                torch.cat(
                    [
                        reward.compute(flat[i : i + 256], condition, timestep, **kwargs)
                        for i in range(0, len(flat), 256)
                    ]
                )
                .reshape(n, m)
                .mean(dim=1)
            )
            if self.normalization == "zscore" and n > 1:
                raw = (raw - raw.mean()) / raw.std(unbiased=False).clamp_min(1e-6)
            scores.append(reward.weight * raw)
        return torch.stack(scores).sum(dim=0)


class AffineExpressionReward:
    """Evaluate normalized-space diffusion outputs in log-expression units."""

    def __init__(self, reward, mean, scale):
        self.reward, self.mean, self.scale = reward, mean, scale

    def compute(self, x_pred, condition, timestep, **kwargs):
        mean = torch.as_tensor(self.mean, device=x_pred.device, dtype=x_pred.dtype)
        scale = torch.as_tensor(self.scale, device=x_pred.device, dtype=x_pred.dtype)
        controls = kwargs.get("ctrl_cells")
        if controls is not None:
            kwargs["ctrl_cells"] = controls * scale + mean
        return self.reward.compute(x_pred * scale + mean, condition, timestep, **kwargs)
