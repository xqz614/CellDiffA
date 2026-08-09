import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/baselines/run_perturbdiff_released.sh"
CELLDIFFA_LAUNCHER = ROOT / "scripts/baselines/run_celldiffa_replogle.sh"


def _run_launcher(tmp_path: Path, variant: str) -> list[str]:
    data_root = tmp_path / "data"
    perturb_root = tmp_path / "PerturbDiff"
    upstream_entrypoint = perturb_root / "src/apps/run/rawdata_diffusion_sampling.py"
    upstream_entrypoint.parent.mkdir(parents=True)
    upstream_entrypoint.touch()

    checkpoint_dir = data_root / "checkpoints/PerturbDiff_release_ckpt"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_name = (
        "from_scratch_replogle.ckpt" if variant == "scratch" else "finetuned_replogle.ckpt"
    )
    (checkpoint_dir / checkpoint_name).touch()

    perturb_data = data_root / "PerturbDiff_data"
    perturb_data.mkdir()
    selected_genes = perturb_data / "selected_genes"
    selected_genes.mkdir()
    (selected_genes / "merged_pbmc_tahoe_rep_cellxgene_genes_mapped.pkl").touch()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "args.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["CELLDIFFA_DATA_ROOT"] = str(data_root)
    env["CAPTURE_ARGS"] = str(capture)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "replogle",
            variant,
            str(tmp_path / "output"),
            "1",
            str(perturb_root),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    return capture.read_text(encoding="utf-8").splitlines()


def test_replogle_scratch_uses_2000_hvg_space(tmp_path):
    args = _run_launcher(tmp_path, "scratch")
    assert "data=replogle_finetune" in args
    assert "model.input_dim=2000" in args
    assert "data.pad_length=2000" in args
    assert "data.embed_key=X_hvg" in args


def test_replogle_finetuned_uses_12626_pretraining_gene_space(tmp_path):
    args = _run_launcher(tmp_path, "finetuned")
    assert "data=replogle_finetune" in args
    assert "model.input_dim=12626" in args
    assert "data.pad_length=12626" in args
    assert "data.embed_key=X" in args
    assert not any(arg.startswith("data.skip_tahoe100m=") for arg in args)
    assert not any(arg.startswith("data.skip_pbmc=") for arg in args)


def test_celldiffa_launcher_supports_resumable_worker_and_smoke_args(tmp_path):
    data_root = tmp_path / "data"
    perturb_root = tmp_path / "PerturbDiff"
    (perturb_root / "src/apps/run").mkdir(parents=True)
    (perturb_root / "src/apps/run/rawdata_diffusion_sampling.py").touch()
    split = perturb_root / "configs/data/perturb_data/replogle.yaml"
    split.parent.mkdir(parents=True)
    split.touch()
    checkpoint = data_root / "checkpoints/PerturbDiff_release_ckpt/from_scratch_replogle.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    perturb_data = data_root / "PerturbDiff_data"
    source = perturb_data / "finetune_data/nadig_processed_data/replogle.h5ad"
    source.parent.mkdir(parents=True)
    source.touch()
    genes = perturb_data / "selected_genes/replogle_real_selected_genes.pkl"
    genes.parent.mkdir(parents=True)
    genes.touch()
    real = tmp_path / "real.h5ad"
    real.parent.mkdir(parents=True, exist_ok=True)
    real.touch()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "args.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n', encoding="utf-8"
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "CELLDIFFA_DATA_ROOT": str(data_root),
            "CAPTURE_ARGS": str(capture),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CELLDIFFA_NUM_PARTICLES": "8",
            "CELLDIFFA_REAL_TEST": str(real),
        }
    )
    subprocess.run(
        [
            "bash",
            str(CELLDIFFA_LAUNCHER),
            "scratch",
            str(tmp_path / "output"),
            "3",
            "2",
            "4",
            "1",
            str(perturb_root),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    args = capture.read_text(encoding="utf-8").splitlines()
    assert "--worker-index" in args and "2" in args
    assert "--num-workers" in args and "4" in args
    assert "--max-groups" in args and "1" in args
    assert "--num-particles" in args and "8" in args
    assert "model.input_dim=2000" in args
