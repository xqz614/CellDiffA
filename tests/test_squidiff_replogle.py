import json
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.baselines import run_squidiff_replogle as runner


def install_tiny_upstream(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)

    class Diffusion:
        def training_losses(self, model, x, t, model_kwargs):
            return {"loss": ((model.encoder(x) - x) ** 2).mean(dim=1)}

    class Sampler:
        def __init__(self, diffusion):
            pass

        def sample(self, count, device):
            return torch.zeros(count, dtype=torch.long, device=device), torch.ones(
                count, device=device
            )

    root = types.ModuleType("Squidiff")
    diffusion = types.ModuleType("Squidiff.diffusion")
    root.diffusion = diffusion
    resample = types.ModuleType("Squidiff.resample")
    resample.UniformSampler = Sampler
    script_util = types.ModuleType("Squidiff.script_util")
    script_util.model_and_diffusion_defaults = lambda: {}
    script_util.create_model_and_diffusion = lambda **kwargs: (Model(), Diffusion())
    for module in (root, diffusion, resample, script_util):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: runner.REVISION)


def make_references(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    var = pd.DataFrame(index=["G1", "G2"])
    train = ad.AnnData(
        np.ones((4, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"gene": ["ctrl", "A", "ctrl", "A"], "cell_line": ["source"] * 4},
            index=["c0", "c1", "c2", "c3"],
        ),
        var=var,
    )
    validation = ad.AnnData(
        np.ones((2, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"gene": ["ctrl", "V"], "cell_line": ["target", "target"]},
            index=["v0", "v1"],
        ),
        var=var.copy(),
    )
    train.write_h5ad(root / "train.h5ad")
    validation.write_h5ad(root / "validation.h5ad")
    split = tmp_path / "split.yaml"
    split.write_text(
        "pert_col: gene\ncontrol_pert: ctrl\ncell_line_key: cell_line\n"
        "holdout_celltype: [target]\nholdout_pert:\n"
        "  validation: [V]\n  test: [T]\n"
    )
    return root, split


def test_train_only_never_reads_test_and_can_resume(monkeypatch, tmp_path):
    install_tiny_upstream(monkeypatch)
    root, split = make_references(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(runner, "read_h5ad_obs", lambda *a: pytest.fail("Test labels accessed"))
    actual_read = runner.ad.read_h5ad

    def read_only_training(path, *args, **kwargs):
        assert Path(path).name in {"train.h5ad", "validation.h5ad"}
        return actual_read(path, *args, **kwargs)

    monkeypatch.setattr(runner.ad, "read_h5ad", read_only_training)
    argv = [
        "runner",
        "--stage",
        "train",
        "--device",
        "cpu",
        "--num-threads",
        "1",
        "--iterations",
        "2",
        "--validation-every",
        "1",
        "--batch-size",
        "2",
        "--reference-dir",
        str(root),
        "--split-config",
        str(split),
        "--output-dir",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    checkpoint_hash = runner.sha256_file(output / "best.pt")
    progress = json.loads((output / "training_progress.json").read_text())
    assert progress["status"] == "training_complete"
    assert progress["completed_steps"] == 2
    assert not progress["test_responses_accessed"]
    assert not (output / "predictions.h5ad").exists()
    assert not (output / "prediction_config.json").exists()
    runner.main()
    assert runner.sha256_file(output / "best.pt") == checkpoint_hash
    # Content hashes reject a changed training artifact when resuming.
    changed = actual_read(root / "train.h5ad")
    changed.X[:] = 2
    changed.write_h5ad(root / "train.h5ad")
    with pytest.raises(ValueError, match="settings changed"):
        runner.main()


def test_unknown_prediction_policy_is_not_silent(monkeypatch, tmp_path):
    from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit

    split = PerturbDiffSplit(
        "gene", "ctrl", "cell_line", None, ("target",), frozenset({"V"}), frozenset({"T"})
    )
    target = pd.DataFrame({"gene": ["T"], "cell_line": ["target"]})
    train = pd.DataFrame({"gene": ["ctrl", "A"]})
    monkeypatch.setattr(runner, "read_h5ad_obs", lambda *a: target)
    with pytest.raises(ValueError, match="explicit policy required"):
        runner.prediction_metadata(tmp_path, train, split, "error")
    obs, unknown = runner.prediction_metadata(tmp_path, train, split, "zero_shift")
    assert obs.equals(target)
    assert unknown == ["T"]


def test_prediction_requires_completed_training(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--stage", "predict", "--split-config", "unused", "--output-dir", str(tmp_path)],
    )
    with pytest.raises(ValueError, match="completed training"):
        runner.main()


def test_latent_shift_uses_training_context_controls():
    obs = pd.DataFrame(
        {"gene": ["ctrl", "A", "ctrl", "A"], "cell_line": ["one", "one", "two", "two"]}
    )
    shifts = runner.latent_shifts(np.array([[10.0], [13.0], [20.0], [25.0]]), obs, "ctrl")
    np.testing.assert_allclose(shifts["A"], [4.0])
