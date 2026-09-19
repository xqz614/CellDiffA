"""Load and apply the dataset splits released with PerturbDiff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _as_values(value) -> list:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [value]
    return list(value)


@dataclass(frozen=True)
class PerturbDiffSplit:
    """The fields needed to reproduce one PerturbDiff holdout split."""

    pert_col: str
    control_pert: str
    context_col: str
    batch_col: str | None
    holdout_contexts: tuple[str, ...]
    validation_perts: frozenset[str]
    test_perts: frozenset[str]

    @classmethod
    def from_yaml(cls, path: str | Path, *, split_axis: str = "context") -> "PerturbDiffSplit":
        with Path(path).open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, dict):
            raise ValueError(f"PerturbDiff split config must be a mapping: {path}")

        context_col = values.get("cell_line_key") or values.get("cell_type_key")
        batch_col = values.get("perturbseq_batch_col")
        if not context_col:
            raise ValueError("PerturbDiff split config has no cell-line/cell-type key.")
        if split_axis == "context":
            holdouts = values.get("holdout_celltype")
        elif split_axis == "batch":
            holdouts = values.get("holdout_batches")
        else:
            raise ValueError("split_axis must be 'context' or 'batch'.")
        if not holdouts:
            raise ValueError(f"PerturbDiff config has no holdouts for split axis {split_axis!r}.")

        holdout_pert = values.get("holdout_pert") or {}
        holdouts = _as_values(holdouts)
        validation = _as_values(holdout_pert.get("validation"))
        testing = _as_values(holdout_pert.get("test"))
        overlap = {str(value) for value in validation} & {str(value) for value in testing}
        if overlap:
            raise ValueError(
                "PerturbDiff validation and test perturbations overlap: "
                f"{sorted(overlap)[:10]}"
            )
        return cls(
            pert_col=str(values["pert_col"]),
            control_pert=str(values["control_pert"]),
            context_col=str(context_col),
            batch_col=str(batch_col) if batch_col else None,
            holdout_contexts=tuple(str(value) for value in holdouts),
            validation_perts=frozenset(str(value) for value in validation),
            test_perts=frozenset(str(value) for value in testing),
        )

    def masks(
        self,
        obs: pd.DataFrame,
        *,
        split_axis: str = "context",
    ) -> dict[str, np.ndarray]:
        """Return mutually exclusive train/validation/test row masks."""
        axis_col = self.context_col if split_axis == "context" else self.batch_col
        if not axis_col:
            raise ValueError(f"No column is configured for split axis {split_axis!r}.")
        required = {self.pert_col, axis_col}
        missing = required - set(obs.columns)
        if missing:
            raise ValueError(f"Source data is missing split columns: {sorted(missing)}")

        labels = obs[self.pert_col].astype(str).to_numpy()
        axes = obs[axis_col].astype(str).to_numpy()
        in_holdout = np.isin(axes, self.holdout_contexts)
        validation = in_holdout & np.isin(labels, tuple(self.validation_perts))
        testing = in_holdout & np.isin(labels, tuple(self.test_perts))
        training = ~(validation | testing)
        if np.any(validation & testing):
            raise AssertionError("PerturbDiff validation and test masks overlap.")
        return {"train": training, "validation": validation, "test": testing}

    def validate_real_test(self, real, *, require_single_holdout: bool = True) -> None:
        """Reject a real-test H5AD that does not match this released split."""
        self.validate_reference(real, require_single_holdout=require_single_holdout)

    def validate_reference(
        self, real, *, split_name: str = "test", require_single_holdout: bool = True
    ) -> None:
        """Validate validation and test references without interchanging them."""
        if split_name not in {"validation", "test"}:
            raise ValueError("split_name must be validation or test.")
        expected_perts = self.test_perts if split_name == "test" else self.validation_perts
        required = {self.pert_col, self.context_col}
        missing = required - set(real.obs.columns)
        if missing:
            raise ValueError(f"Real test is missing obs columns: {sorted(missing)}")
        labels = real.obs[self.pert_col].astype(str)
        treated = labels != self.control_pert
        observed_perts = set(labels[treated])
        unexpected = observed_perts - set(expected_perts)
        if unexpected:
            raise ValueError(
                f"Reference contains perturbations outside the PerturbDiff {split_name} split: "
                f"{sorted(unexpected)[:10]}"
            )
        if not observed_perts:
            raise ValueError("Real test contains no treated perturbations.")
        observed_contexts = set(real.obs.loc[treated, self.context_col].astype(str))
        unexpected_contexts = observed_contexts - set(self.holdout_contexts)
        if unexpected_contexts:
            raise ValueError(
                "Real test contains contexts outside the PerturbDiff holdout: "
                f"{sorted(unexpected_contexts)}"
            )
        if require_single_holdout and len(observed_contexts) != 1:
            raise ValueError(
                "GEARS-Replogle expects exactly one held-out test context; "
                f"found {sorted(observed_contexts)}."
            )
