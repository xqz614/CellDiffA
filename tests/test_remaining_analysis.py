"""Tiny synthetic fixtures verify reports; these are not experimental results."""

import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from scripts.baselines import analyze_adacell_experiments as analysis


def test_backbone_pairs_label_unmatched_budget_and_prefer_matched_reference():
    tables = {"squidiff_vanilla": None, "squidiff_adacell16": None}
    assert analysis.backbone_comparisons(tables) == [
        ("Squidiff", "squidiff_adacell16", "squidiff_vanilla", False)
    ]
    tables["squidiff_random16"] = None
    assert analysis.backbone_comparisons(tables) == [
        ("Squidiff", "squidiff_adacell16", "squidiff_random16", True)
    ]


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
        "accuracy_base_drift",
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


def test_strict_analysis_requires_all_controls_before_writing(tmp_path, monkeypatch):
    real_dir = tmp_path / "results/replogle/reference"
    real_dir.mkdir(parents=True)
    data = ad.AnnData(
        np.zeros((1, 2)),
        obs=pd.DataFrame(dict(gene=["non-targeting"], cell_line=["A"]), index=["c"]),
    )
    data.write_h5ad(real_dir / "real.h5ad")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(repo=str(tmp_path), output_root=str(tmp_path), lanes=[])))
    output = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze",
            "--plan",
            str(plan),
            "--main-pred",
            str(tmp_path / "pred.h5ad"),
            "--main-metrics",
            str(tmp_path),
            "--outdir",
            str(output),
            "--perturbdiff-only",
            "--require-controls",
        ],
    )
    with pytest.raises(ValueError, match="Missing required controls"):
        analysis.main()
    assert not output.exists()


def test_strict_budget_audit_distinguishes_eight_particles_and_rejects_wrong_prior(
    tmp_path, monkeypatch
):
    import copy

    from celldiffa.benchmark.artifacts import sha256_file
    from scripts.baselines import audit_replogle_steering_budget as budget

    config = {key: "fixed" for key in budget.MATCHED_SETTINGS}
    config.update(
        variant="scratch",
        evaluation_split="test",
        alpha=1.0,
        seed=42,
        num_particles=16,
        alignment_mode="smc",
        reward_normalization="zscore",
        reward_weights=[1, 1, 1],
        ess_threshold=0.5,
        prior_ridge=1,
        top_de=20,
        anchor_bandwidth=1,
        anchor_estimator="fixed",
        perturbation_embeddings_sha256="fixed",
    )
    summary = dict(
        sampling_coverage_complete=True, groups={0: ["A", 3, 32, 51200]}, denoised_cell_steps=51200
    )
    records, states = {}, {}
    for name in ("scratch_main", *analysis.CONTROL_NAMES):
        path = tmp_path / name
        path.mkdir()
        prediction = path / "predictions.h5ad"
        prediction.write_text("fixture")
        records[name] = dict(output=path, prediction=prediction)
        cfg, detail = copy.deepcopy(config), copy.deepcopy(summary)
        if name == "scratch_random16":
            cfg["alignment_mode"] = "random"
        elif name == "scratch_best16":
            cfg["alignment_mode"] = "best_of_n"
        elif name == "scratch_cellwise16":
            cfg["reward_unit"] = "cell"
        elif name == "scratch_particles8":
            cfg["num_particles"] = 8
            detail["groups"][0][3] //= 2
            detail["denoised_cell_steps"] //= 2
        states[path / "shards"] = (cfg, detail)
    marker = records["scratch_mean_correction"]["output"] / "provenance.json"
    marker.write_text(
        json.dumps(
            dict(
                source_prediction_sha256=sha256_file(records["scratch_random16"]["prediction"]),
                test_response_values_used_for_correction=False,
            )
        )
    )
    monkeypatch.setattr(budget, "read_run", lambda path: states[path])
    actual = analysis.audit_controls(records)
    assert actual["scratch_random16"]["full_budget_match_verified"]
    assert not actual["scratch_particles8"]["full_budget_match_verified"]
    assert actual["scratch_particles8"]["denoised_cell_steps_ratio"] == 0.5
    states[records["scratch_best16"]["output"] / "shards"][0]["prior_ridge"] = 2
    with pytest.raises(ValueError, match="prior_ridge"):
        analysis.audit_controls(records)
