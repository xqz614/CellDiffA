"""Tiny synthetic fixtures verify reports; these are not experimental results."""

import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from scripts.baselines import analyze_adacell_experiments as analysis


def test_analysis_plots_and_rejects_mismatched_metrics(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    reference = tmp_path / "results/replogle/reference"
    reference.mkdir(parents=True)
    labels = np.repeat(["non-targeting", "A", "B", "C"], 5)
    rng = np.random.default_rng(7)
    real = ad.AnnData(
        rng.uniform(0.1, 2, (20, 6)).astype(np.float32),
        obs=pd.DataFrame(
            dict(gene=labels, cell_line=["context"] * 20),
            index=[f"c{i}" for i in range(20)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(6)]),
    )
    real.write_h5ad(reference / "real.h5ad")
    root = tmp_path / "remaining"
    jobs = [dict(id="scratch_random16", kind="perturbdiff"), dict(id="pending", kind="squidiff")]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(repo=str(tmp_path), output_root=str(root), lanes=[jobs])))
    for name, error in [("scratch_main", 0.01), ("scratch_random16", 0.03)]:
        output = root / "runs" / name
        metrics = root / "metrics" / name
        output.mkdir(parents=True)
        metrics.mkdir(parents=True)
        pred = real.copy()
        pred.X[labels != "non-targeting"] += error
        pred.write_h5ad(output / "celldiffa_scratch.h5ad")
        rows = []
        for pert in ("A", "B", "C"):
            actual = real.X[labels == pert].mean(0)
            predicted = pred.X[labels == pert].mean(0)
            rows.append(
                dict(
                    perturbation=pert,
                    MSE=float(np.square(actual - predicted).mean()),
                    R2=r2_score(actual, predicted),
                    PDCorr=0.5,
                    PDS_cos=0.5,
                    DEOver=0.2,
                )
            )
        pd.DataFrame(rows).to_csv(metrics / "perturbdiff_metrics_per_perturbation.csv", index=False)
    args = [
        "analyze",
        "--plan",
        str(plan),
        "--main-pred",
        str(root / "runs/scratch_main/celldiffa_scratch.h5ad"),
        "--main-metrics",
        str(root / "metrics/scratch_main"),
        "--outdir",
        str(tmp_path / "figures"),
    ]
    monkeypatch.setattr(sys, "argv", args)
    analysis.main()
    manifest = json.loads((tmp_path / "figures/analysis_manifest.json").read_text())
    assert manifest["missing_jobs"] == ["pending"] and not manifest["protocol_complete"]
    for name in (
        "contrast_diagnosis",
        "population_controls",
        "accuracy_diversity",
        "backbone_transfer",
    ):
        assert (tmp_path / f"figures/{name}.pdf").stat().st_size > 100
        assert (tmp_path / f"figures/{name}.png").stat().st_size > 100
    seeds = pd.read_csv(tmp_path / "figures/paired_seed_deltas.csv")
    assert set(seeds.seed) == {42}
    table_path = root / "metrics/scratch_main/perturbdiff_metrics_per_perturbation.csv"
    table = pd.read_csv(table_path)
    table["MSE"] += 0.1
    table.to_csv(table_path, index=False)
    monkeypatch.setattr(sys, "argv", args[:-1] + [str(tmp_path / "bad_report")])
    with pytest.raises(ValueError, match="does not match prediction"):
        analysis.main()
