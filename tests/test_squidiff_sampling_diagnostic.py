import hashlib
import json
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from celldiffa.benchmark.artifacts import sha256_file
from scripts.server import diagnose_squidiff_sampling as diagnostic


def test_condition_selection_uses_metadata_and_training_support():
    train = pd.DataFrame({"gene": ["ctrl", "A", "B"]})
    val = pd.DataFrame({"gene": ["B", "C", "A", "ctrl"], "cell_line": ["x"] * 4})
    assert diagnostic.choose_conditions(train, val, "ctrl", 1) == [("A", "x")]
    val.loc[3, "cell_line"] = "other"
    assert diagnostic.choose_conditions(train, val, "ctrl", 2) == []


def test_pinned_native_multistep_matches_complete_unguided_engine():
    root = Path(__file__).parents[1] / "external/Squidiff"
    if not root.exists():
        pytest.skip("Pinned upstream source optional in CI")
    sys.path.insert(0, str(root))
    from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        kwargs = model_and_diffusion_defaults()
        kwargs.update(gene_size=2, output_dim=2, use_encoder=True, timestep_respacing="ddim10")
        model, diffusion = create_model_and_diffusion(**kwargs)
        model.eval()
        noise = torch.randn(3, 2)
        with torch.no_grad():
            condition = model.encoder(torch.ones_like(noise))
        native = diagnostic.native_sample(model, diffusion, noise, condition)
        adapter = diagnostic.adapter_sample(model, diffusion, noise, condition, seed=42)
        assert torch.isfinite(native).all()
        torch.testing.assert_close(native, adapter, rtol=1e-6, atol=1e-6)
    finally:
        torch.set_num_threads(old_threads)


def test_diagnostic_cli_never_reads_test_or_retrains(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)

    class Diffusion:
        def __init__(self, steps):
            self.num_timesteps = steps
            self.timestep_map = list(range(steps))

        def ddim_sample(self, model, x, t, **kwargs):
            assert kwargs["clip_denoised"] is False and kwargs["eta"] == 0
            values = x * 0.99 + 0.001 * kwargs["model_kwargs"]["z_mod"]
            return {"sample": values, "pred_xstart": values}

        def ddim_sample_loop(self, model, shape, noise, **kwargs):
            x = noise.clone()
            for t in reversed(range(self.num_timesteps)):
                x = self.ddim_sample(model, x, t, **kwargs)["sample"]
            return x

    root, ref = tmp_path, tmp_path / "results/replogle/reference"
    ref.mkdir(parents=True)
    var = pd.DataFrame(index=["G1", "G2"])
    train = ad.AnnData(
        np.ones((4, 2), dtype=np.float32),
        var=var.copy(),
        obs=pd.DataFrame(
            {"gene": ["non-targeting", "V"] * 2, "cell_line": ["source"] * 4},
            index=[f"t{i}" for i in range(4)],
        ),
    )
    val = ad.AnnData(
        np.ones((4, 2), dtype=np.float32),
        var=var.copy(),
        obs=pd.DataFrame(
            {"gene": ["non-targeting", "non-targeting", "V", "V"], "cell_line": ["target"] * 4},
            index=[f"v{i}" for i in range(4)],
        ),
    )
    train.write_h5ad(ref / "train.h5ad")
    val.write_h5ad(ref / "validation.h5ad")
    split = root / "external/PerturbDiff/configs/data/perturb_data/replogle.yaml"
    split.parent.mkdir(parents=True)
    split.write_text(
        "pert_col: gene\ncontrol_pert: non-targeting\ncell_line_key: cell_line\n"
        "holdout_celltype: [target]\nholdout_pert:\n"
        "  validation: [V]\n  test: [T]\n"
    )
    model_dir = root / "model"
    model_dir.mkdir()
    checkpoint = model_dir / "best.pt"
    torch.save(Model().state_dict(), checkpoint)
    original = checkpoint.read_bytes()
    config = dict(
        revision=diagnostic.REVISION,
        smoke=False,
        train_sha256=sha256_file(ref / "train.h5ad"),
        validation_sha256=sha256_file(ref / "validation.h5ad"),
        split_sha256=sha256_file(split),
        ordered_genes_sha256=hashlib.sha256(b"G1\nG2").hexdigest(),
    )
    (model_dir / "run_config.json").write_text(json.dumps(config))
    (model_dir / "training_progress.json").write_text(json.dumps({"status": "training_complete"}))
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            dict(repo=str(root), data_root=str(root / "data"), squidiff_checkpoint=str(checkpoint))
        )
    )
    fake = types.ModuleType("Squidiff.script_util")
    fake.model_and_diffusion_defaults = lambda: {"diffusion_steps": 1000}
    fake.create_model_and_diffusion = lambda **kw: (
        Model(),
        Diffusion(100 if kw["timestep_respacing"] else 1000),
    )
    monkeypatch.setitem(sys.modules, "Squidiff", types.ModuleType("Squidiff"))
    monkeypatch.setitem(sys.modules, "Squidiff.script_util", fake)
    monkeypatch.setattr(diagnostic.subprocess, "check_output", lambda *a, **kw: diagnostic.REVISION)
    actual_read = ad.read_h5ad

    def only_training_validation(path, *args, **kw):
        assert Path(path).name in {"train.h5ad", "validation.h5ad"}
        return actual_read(path, *args, **kw)

    monkeypatch.setattr(diagnostic.ad, "read_h5ad", only_training_validation)
    out = root / "diagnostic"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnostic",
            "--config",
            str(plan),
            "--device",
            "cpu",
            "--cells",
            "2",
            "--groups",
            "1",
            "--output-dir",
            str(out),
        ],
    )
    diagnostic.main()
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "diagnostic_complete" and not report["test_responses_accessed"]
    assert len(report["cases"]) == 2
    for case in report["cases"]:
        assert case["native_adapter_100"]["allclose"]
        assert set(case["outputs"]) == {"native100", "native1000", "adapter100"}
    assert original == checkpoint.read_bytes()
    assert not (out / "predictions.h5ad").exists()
