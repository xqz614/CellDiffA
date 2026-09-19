"""Vectorized reduction of Scouter's official condition-balanced gene loss."""

import numpy as np
import torch


class ScouterGroupedLoss:
    """Same equation as scouter._utils.gears_loss, fewer GPU kernel launches."""

    def __init__(self, nonzero_idx_dict, genes, device, gamma=0.0, direction_weight=0.5):
        self.names = {name: index for index, name in enumerate(sorted(nonzero_idx_dict))}
        masks = np.zeros((len(self.names), genes), dtype=np.float32)
        for name, row in self.names.items():
            indices = nonzero_idx_dict[name]
            if not len(indices):
                raise ValueError(f"Empty gene filter for {name}")
            masks[row, indices] = 1.0 / len(indices)
        self.masks = torch.tensor(masks, device=device)
        self.gamma, self.direction_weight = gamma, direction_weight

    def __call__(self, prediction, target, control, groups):
        unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        rows = [self.names[name] for name in groups]
        mask = self.masks[torch.tensor(rows, device=prediction.device)]
        weights = torch.tensor(
            1.0 / (len(unique) * counts[inverse]), device=prediction.device, dtype=prediction.dtype
        )
        loss = (target - prediction).abs().pow(2 + self.gamma)
        loss = (
            loss
            + self.direction_weight
            * (torch.sign(target - control) - torch.sign(prediction - control)).square()
        )
        return (loss * mask * weights[:, None]).sum()
