"""Synthetic unit fixtures for prior interventions, not manuscript results."""

import importlib.util
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from celldiffa.benchmark.replogle_priors import (
    _shuffled_prior_donors,
    _subsample_training_mask,
    compute_replogle_training_priors,
    validate_prior_robustness,
)


@pytest.fixture
def prior_fixture(tmp_path):
    labels, contexts, values = [], [], []
    for context, baseline in [("A", 1), ("C", 10), ("B", 20)]:
        for offset in [-0.5, 0.5]:
            labels.append("control")
            contexts.append(context)
            values.append([baseline + offset] * 3)
        if context != "B":
            for perturbation, column in [("P", 0), ("Q", 1)]:
                for magnitude in range(1, 9):
                    response = np.full(3, baseline, dtype=float)
                    response[column] += magnitude
                    labels.append(perturbation)
                    contexts.append(context)
                    values.append(response)
    for perturbation in ["P", "Q", "U", "V"]:
        labels.append(perturbation)
        contexts.append("B")
        values.append([100_000.0, 500_000.0, 900_000.0])
    obs = pd.DataFrame(
        {"gene": labels, "context": contexts}, index=[f"c{i}" for i in range(len(labels))]
    )
    data = ad.AnnData(
        X=np.asarray(values, dtype=np.float32), obs=obs, var=pd.DataFrame(index=["g0", "g1", "g2"])
    )
    data.obsm["X_hvg"] = data.X.copy()
    source = tmp_path / "source.h5ad"
    data.write_h5ad(source)
    split = tmp_path / "split.yaml"
    split.write_text(
        yaml.safe_dump(
            {
                "pert_col": "gene",
                "control_pert": "control",
                "cell_line_key": "context",
                "holdout_celltype": ["B"],
                "holdout_pert": {"validation": ["V"], "test": ["P", "Q", "U"]},
            }
        )
    )
    common = {
        "source": source,
        "split_path": split,
        "genes": ["g0", "g1", "g2"],
        "perturbation_embeddings": {
            "P": np.array([1.0, 0.0]),
            "Q": np.array([0.0, 1.0]),
            "U": np.array([0.8, 0.2]),
        },
        "embedding_signature": "synthetic-unit-test-v1",
        "top_k": 1,
    }
    return data, common


@pytest.mark.parametrize("fraction", [0.5, 0.25])
def test_subsampling_preserves_controls_and_strata_and_excludes_heldout(fraction):
    labels = np.array(["control"] * 2 + ["P"] * 8 + ["Q"] * 8 + ["P"])
    contexts = np.array(["A"] * 10 + ["C"] * 8 + ["B"])
    train = np.array([True] * 18 + [False])
    selected = _subsample_training_mask(
        train, labels, contexts, "control", fraction=fraction, seed=42
    )
    assert selected[:2].all()
    assert not selected[-1]
    assert selected[2:10].sum() == int(8 * fraction)
    assert selected[10:18].sum() == int(8 * fraction)
    again = _subsample_training_mask(train, labels, contexts, "control", fraction=fraction, seed=42)
    np.testing.assert_array_equal(selected, again)
    np.testing.assert_array_equal(train, np.array([True] * 18 + [False]))


def test_subsampling_retains_at_least_one_cell_per_stratum():
    selected = _subsample_training_mask(
        np.ones(3, dtype=bool),
        np.array(["control", "P", "Q"]),
        np.array(["A"] * 3),
        "control",
        fraction=0.25,
        seed=42,
    )
    assert selected.all()


@pytest.mark.parametrize("fraction,expected", [(0.5, 8), (0.25, 4)])
def test_subsample_refits_train_only_and_is_chunk_independent(prior_fixture, fraction, expected):
    data, common = prior_fixture
    options = {"prior_mode": "subsample", "prior_fraction": fraction, "prior_seed": 43}
    first = compute_replogle_training_priors(**common, **options, chunk_size=3)
    assert first.counts == {"P": expected, "Q": expected, "U": 0}
    assert first.sources["U"] == "genept_dual_ridge"
    heldout = (data.obs["context"] == "B") & (data.obs["gene"] != "control")
    data.obsm["X_hvg"][heldout.to_numpy()] = np.nan
    data.write_h5ad(common["source"])
    second = compute_replogle_training_priors(**common, **options, chunk_size=11)
    for perturbation in first.shifts:
        np.testing.assert_allclose(first.shifts[perturbation], second.shifts[perturbation])
        assert np.isfinite(second.shifts[perturbation]).all()
        assert first.de_genes[perturbation] == [
            common["genes"][np.argmax(np.abs(first.shifts[perturbation]))]
        ]


def test_shuffle_deranges_priors_and_recomputes_signature_genes(prior_fixture):
    _, common = prior_fixture
    original = compute_replogle_training_priors(**common)
    shuffled = compute_replogle_training_priors(**common, prior_mode="shuffle", prior_seed=44)
    donors = _shuffled_prior_donors(original.shifts, 44)
    assert set(donors) == set(donors.values())
    for target, donor in donors.items():
        assert target != donor
        np.testing.assert_array_equal(shuffled.shifts[target], original.shifts[donor])
        assert shuffled.de_genes[target] == original.de_genes[donor]
        assert shuffled.counts[target] == original.counts[donor]
        assert shuffled.sources[target] == f"shuffled:{donor}:{original.sources[donor]}"
    assert _shuffled_prior_donors(reversed(list(original.shifts)), 44) == donors
    with pytest.raises(ValueError, match="at least two"):
        _shuffled_prior_donors(["P"], 42)


def test_default_cache_compatible_but_robustness_cache_never_false_hits(prior_fixture, tmp_path):
    _, common = prior_fixture
    cache = tmp_path / "priors.npz"
    original = compute_replogle_training_priors(**common, cache_path=cache)
    with np.load(cache) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
    assert metadata["format_version"] == 3
    assert "prior_mode" not in metadata
    sampled = compute_replogle_training_priors(
        **common, cache_path=cache, prior_mode="subsample", prior_fraction=0.5
    )
    assert sampled.counts["P"] == 8 != original.counts["P"]
    with np.load(cache) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
    assert metadata["format_version"] == 4
    assert (metadata["prior_mode"], metadata["prior_fraction"], metadata["prior_seed"]) == (
        "subsample",
        0.5,
        42,
    )
    cached = compute_replogle_training_priors(
        **common, cache_path=cache, prior_mode="subsample", prior_fraction=0.5
    )
    np.testing.assert_array_equal(cached.shifts["P"], sampled.shifts["P"])
    restored = compute_replogle_training_priors(**common, cache_path=cache)
    assert restored.counts == original.counts


@pytest.mark.parametrize(
    "mode,fraction,seed",
    [
        ("bad", 1, 42),
        ("full", 0.5, 42),
        ("shuffle", 0.25, 42),
        ("subsample", 0, 42),
        ("subsample", np.nan, 42),
        ("subsample", 1.1, 42),
        ("full", 1, -1),
    ],
)
def test_invalid_prior_options_are_rejected(mode, fraction, seed):
    with pytest.raises(ValueError):
        validate_prior_robustness(mode, fraction, seed)


def test_cli_exposes_prior_options_and_leaves_hydra_conditioning_unchanged(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/baselines/run_celldiffa_replogle.py"
    spec = importlib.util.spec_from_file_location("prior_robustness_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = [
        "--source",
        "source",
        "--real-test",
        "real",
        "--split-config",
        "split",
        "--selected-genes",
        "genes",
        "--perturbation-embeddings",
        "emb",
        "--prior-cache",
        "cache",
        "--shard-root",
        "shards",
        "--output",
        "out",
        "--variant",
        "scratch",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            *required,
            "--prior-mode",
            "subsample",
            "--prior-fraction",
            "0.25",
            "--prior-seed",
            "44",
            "--top-de",
            "10",
            "sampling.eta=0.0",
        ],
    )
    args, overrides = module.parse_args()
    assert (args.prior_mode, args.prior_fraction, args.prior_seed, args.top_de) == (
        "subsample",
        0.25,
        44,
        10,
    )
    assert overrides == ["sampling.eta=0.0"]
