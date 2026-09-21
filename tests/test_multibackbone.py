import json
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from baselines.adapter_squidiff import SquidiffSampler
from baselines.conditional_ddpm import ConditionalDDPM, ConditionalDDPMSampler
from celldiffa.benchmark.backbone_experiments import plan_groups
from celldiffa.rewards import CompositeReward, TranscriptomicReward
from celldiffa.rewards.cellwise import IndependentCellReward
from celldiffa.smc import SMCConfig, SMCEngine
from scripts.baselines.replogle_mean_correction import correct_mean


def test_cellwise_control_changes_objective_not_input_population():
    term = TranscriptomicReward({"A": ["G"]}, {"A": np.array([1.0])}, np.array([0.0]), ["G"])
    values = torch.tensor([[[0.0], [2.0]], [[1.0], [1.0]]])
    joint = CompositeReward([term], normalization="none").compute(values, "A", 1)
    separate = IndependentCellReward([term], normalization="none").compute(values, "A", 1)
    assert torch.allclose(joint, torch.zeros(2))
    assert separate.tolist() == [-1.0, 0.0]
    assert torch.equal(
        separate,
        IndependentCellReward([term], normalization="none").compute(values.flip(1), "A", 1),
    )


def test_squidiff_adapter_exactly_delegates_native_kernel():
    class Diffusion:
        num_timesteps = 10

        def ddim_sample(self, model, x, t, **kwargs):
            assert kwargs["clip_denoised"] is False
            assert kwargs["eta"] == 0
            return dict(sample=x + kwargs["model_kwargs"]["z_mod"], pred_xstart=x * 2)

    model = torch.nn.Linear(2, 2)
    adapter = SquidiffSampler(model, Diffusion())
    x = torch.ones(3, 2)
    result = adapter.denoise_step(x, torch.ones(3, dtype=torch.long), {"z_mod": x * 3})
    assert torch.equal(result["x_prev"], x * 4)
    assert torch.equal(result["x0_pred"], x * 2)
    assert not any(p.requires_grad for p in model.parameters())


def test_native_squidiff_single_step_smoke():
    root = Path(__file__).parents[1] / "external/Squidiff"
    if not root.exists():
        pytest.skip("Pinned external Squidiff checkout is optional in CI")
    sys.path.insert(0, str(root))
    from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    kwargs = model_and_diffusion_defaults()
    kwargs.update(gene_size=2, output_dim=2, use_encoder=True, timestep_respacing="ddim10")
    model, diffusion = create_model_and_diffusion(**kwargs)
    model.eval()
    x = torch.tensor([[0.2, 0.3], [0.5, 0.9]])
    with torch.no_grad():
        z = model.encoder(x)
        expected = diffusion.ddim_sample(
            model, x, torch.tensor([9, 9]), model_kwargs={"z_mod": z}, clip_denoised=False, eta=0
        )
        result = SquidiffSampler(model, diffusion).denoise_step(
            x, torch.tensor([9, 9]), {"z_mod": z}
        )
    assert torch.allclose(result["x_prev"], expected["sample"])
    assert torch.allclose(result["x0_pred"], expected["pred_xstart"])


def test_conditional_sampler_terminal_and_shared_engine():
    model = ConditionalDDPM(2, 3, width=16, depth=1, timesteps=10)
    sampler = ConditionalDDPMSampler(model, sampling_steps=5)
    condition = {"descriptor": torch.ones(1, 3), "control_mean": torch.ones(1, 2)}
    x = torch.randn(2, 2)
    result = sampler.denoise_step(x, torch.zeros(2, dtype=torch.long), condition)
    assert torch.allclose(result["x_prev"], result["x0_pred"])
    reward = SimpleNamespace(compute=lambda x_pred, **kw: -x_pred.square().mean((1, 2)))
    for mode, particles in [("random", 1), ("random", 4), ("best_of_n", 4), ("smc", 4)]:
        config = SMCConfig(
            num_particles=particles,
            cells_per_particle=3,
            device="cpu",
            alignment_mode=mode,
            batch_size_per_step=5,
        )
        generated = SMCEngine(sampler, reward, config).sample_with_alignment(
            "A", condition, ctrl_cells=torch.ones(3, 2)
        )
        assert generated["samples"].shape == (3, 2)
        assert torch.isfinite(generated["samples"]).all()
        assert generated["denoised_cell_steps"] == particles * 3 * 5
        if mode != "smc":
            assert not any(generated["resample_history"])


def test_mean_correction_preserves_order_and_hits_feasible_target():
    x = np.array([[0, 5], [2, 7], [4, 9]], dtype=np.float32)
    result = correct_mean(x, [0.5, -0.1])
    assert np.all(result >= 0)
    np.testing.assert_allclose(result.mean(0), [0.5, 0], atol=1e-6)
    assert np.all(np.diff(result[:, 0]) >= 0)


def fixture_files(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    split = tmp_path / "split.yaml"
    split.write_text(
        "pert_col: gene\ncontrol_pert: non-targeting\ncell_line_key: cell_line\n"
        "holdout_celltype: [target]\nholdout_pert:\n  validation: [V]\n  test: [T]\n"
    )
    labels = (
        ["non-targeting"] * 3
        + ["A"] * 3
        + ["B"] * 3
        + ["non-targeting"] * 4
        + ["V"] * 2
        + ["T"] * 3
    )
    obs = pd.DataFrame(
        {"gene": labels, "cell_line": ["source"] * 9 + ["target"] * 9},
        index=[f"c{i}" for i in range(18)],
    )
    values = np.random.default_rng(7).uniform(0.1, 2, size=(18, 2)).astype(np.float32)
    data = ad.AnnData(
        values, obs=obs, var=pd.DataFrame({"highly_variable": [True, True]}, index=["G1", "G2"])
    )
    data.obsm["X_hvg"] = values.copy()
    source = tmp_path / "source.h5ad"
    data.write_h5ad(source)
    data[:13].write_h5ad(root / "train.h5ad")
    data[9:15].write_h5ad(root / "validation.h5ad")
    data[[9, 10, 11, 12, 15, 16, 17]].write_h5ad(root / "real.h5ad")
    data[9:13].write_h5ad(root / "controls.h5ad")
    genes, embeddings = tmp_path / "genes.pkl", tmp_path / "embeddings.pkl"
    genes.write_bytes(pickle.dumps(["G1", "G2"]))
    embeddings.write_bytes(
        pickle.dumps({"A": [1, 0, 0], "B": [0, 1, 0], "V": [0, 0, 1], "T": [1, 1, 0]})
    )
    return root, split, source, genes, embeddings


def test_tiny_training_prediction_resume_and_full_coverage(tmp_path, monkeypatch):
    from scripts.baselines import run_adacell_backbone as predict
    from scripts.baselines import train_conditional_ddpm_replogle as train

    root, split, source, genes, embeddings = fixture_files(tmp_path)
    model_dir = tmp_path / "model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--reference-dir",
            str(root),
            "--split-config",
            str(split),
            "--embeddings",
            str(embeddings),
            "--output-dir",
            str(model_dir),
            "--device",
            "cpu",
            "--steps",
            "2",
            "--validation-every",
            "1",
            "--width",
            "16",
            "--depth",
            "1",
            "--diffusion-steps",
            "10",
            "--batch-size",
            "4",
            "--num-threads",
            "1",
        ],
    )
    actual_read = ad.read_h5ad

    def no_test_read(path, *args, **kwargs):
        assert Path(path).name in {"train.h5ad", "validation.h5ad"}
        return actual_read(path, *args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(train.ad, "read_h5ad", no_test_read)
        train.main()
        train.main()  # Completed training remains unchanged.
    assert (
        json.loads((model_dir / "training_progress.json").read_text())["status"]
        == "training_complete"
    )
    out = tmp_path / "predictions"
    argv = [
        "predict",
        "--backbone",
        "conditional_ddpm",
        "--checkpoint",
        str(model_dir / "best.pt"),
        "--reference-dir",
        str(root),
        "--source",
        str(source),
        "--split-config",
        str(split),
        "--selected-genes",
        str(genes),
        "--embeddings",
        str(embeddings),
        "--output-dir",
        str(out),
        "--device",
        "cpu",
        "--sampling-steps",
        "5",
        "--population-cells",
        "2",
        "--batch-cells",
        "3",
        "--num-particles",
        "2",
        "--num-threads",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv + ["--max-groups", "1"])
    predict.main()
    assert not (out / "predictions.h5ad").exists()
    monkeypatch.setattr(sys, "argv", argv)
    predict.main()
    result = ad.read_h5ad(out / "predictions.h5ad")
    assert result.shape == (7, 2)
    assert np.isfinite(result.X).all() and (result.X >= 0).all()
    assert json.loads((out / "completion.json").read_text())["complete"]
    monkeypatch.setattr(sys, "argv", argv + ["--alpha", "2"])
    with pytest.raises(ValueError, match="settings differ"):
        predict.main()


def test_groups_do_not_mix_contexts_or_drop_singleton():
    obs = pd.DataFrame({"gene": ["A"] * 4, "cell_line": ["x"] * 3 + ["y"]})
    groups = plan_groups(obs, control="ctrl", population_cells=2)
    assert sum(x["cells"] for x in groups) == 4
    assert [x["sampled_cells"] for x in groups] == [2, 2, 2]


def test_squidiff_full_adapter_requires_explicit_unseen_policy(tmp_path, monkeypatch):
    from celldiffa.benchmark.artifacts import sha256_file
    from scripts.baselines import run_adacell_backbone as predict
    from scripts.baselines.run_squidiff_replogle import REVISION

    root, split, source, genes, embeddings = fixture_files(tmp_path)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 3)

    class Diffusion:
        num_timesteps = 2

        def ddim_sample(self, model, x, t, *, model_kwargs, **kwargs):
            value = x * 0.5 + model_kwargs["z_mod"][:, :2] * 0.1
            return dict(sample=value, pred_xstart=value)

    parent = types.ModuleType("Squidiff")
    diffusion_module = types.ModuleType("Squidiff.diffusion")
    script_util = types.ModuleType("Squidiff.script_util")
    script_util.model_and_diffusion_defaults = lambda: {}
    script_util.create_model_and_diffusion = lambda **kwargs: (Model(), Diffusion())
    parent.diffusion = diffusion_module
    for module in (parent, diffusion_module, script_util):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    import subprocess

    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: REVISION)
    model_dir = tmp_path / "squidiff"
    model_dir.mkdir()
    torch.save(Model().state_dict(), model_dir / "best.pt")
    (model_dir / "run_config.json").write_text(
        json.dumps(
            dict(
                revision=REVISION,
                genes=["G1", "G2"],
                split_sha256=sha256_file(split),
                train_sha256=sha256_file(root / "train.h5ad"),
                smoke=False,
            )
        )
    )
    argv = [
        "predict",
        "--backbone",
        "squidiff",
        "--checkpoint",
        str(model_dir / "best.pt"),
        "--reference-dir",
        str(root),
        "--source",
        str(source),
        "--split-config",
        str(split),
        "--selected-genes",
        str(genes),
        "--embeddings",
        str(embeddings),
        "--output-dir",
        str(tmp_path / "squidiff_predictions"),
        "--device",
        "cpu",
        "--sampling-steps",
        "2",
        "--population-cells",
        "2",
        "--num-particles",
        "2",
        "--num-threads",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="no native condition"):
        predict.main()
    monkeypatch.setattr(sys, "argv", argv + ["--unseen-policy", "ridge"])
    predict.main()
    contract = json.loads((tmp_path / "squidiff_predictions/run_config.json").read_text())
    assert contract["base"]["unsupported_native_conditions"] == ["T"]
    assert contract["base"]["unseen_policy"] == "ridge"
    assert (tmp_path / "squidiff_predictions/predictions.h5ad").is_file()


def test_training_export_preserves_completed_references(tmp_path, monkeypatch):
    from scripts.data import prepare_replogle_training_only as exporter

    root, split, source, genes, _ = fixture_files(tmp_path)
    before = (root / "real.h5ad").read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "--source",
            str(source),
            "--split-config",
            str(split),
            "--selected-genes",
            str(genes),
            "--output",
            str(tmp_path / "new_train.h5ad"),
        ],
    )
    exporter.main()
    exporter.main()
    assert ad.read_h5ad(tmp_path / "new_train.h5ad").n_obs == 13
    assert (root / "real.h5ad").read_bytes() == before
