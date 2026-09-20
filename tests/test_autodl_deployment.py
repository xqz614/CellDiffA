import json
import subprocess
from pathlib import Path

import pytest

from celldiffa.benchmark.artifacts import sha256_file
from scripts.server.replogle_autodl import check_references, ensure_upstream, smoke_command

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("variant,gpu", [("scratch", 0), ("finetuned", 2)])
def test_smoke_is_validation_only_with_unchanged_candidate_budget(variant, gpu, monkeypatch):
    monkeypatch.setenv("CELLDIFFA_NUM_PARTICLES", "1")
    monkeypatch.setenv("CELLDIFFA_ALPHA", "99")
    monkeypatch.setenv("CELLDIFFA_EVALUATION_SPLIT", "test")
    command, env = smoke_command(repo=ROOT, variant=variant, gpu=gpu, output="smoke")
    assert command[-6:] == ["smoke", str(gpu), "0", "1", "1", str(ROOT / "external/PerturbDiff")]
    assert env["CELLDIFFA_DEVICE"] == "cuda:0"
    assert env["CELLDIFFA_NUM_PARTICLES"] == "16"
    assert env["CELLDIFFA_NATIVE_BLOCKS_PER_POPULATION"] == "16"
    assert env["CELLDIFFA_PARTICLE_BATCH_CELLS"] == "1024"
    assert env["CELLDIFFA_EVALUATION_SPLIT"] == "validation"
    assert env["CELLDIFFA_REAL_TEST"].endswith("/reference/validation.h5ad")
    assert env["CELLDIFFA_ALPHA"] == "1.0"
    assert env["CELLDIFFA_ALIGNMENT_MODE"] == "smc"


@pytest.mark.parametrize("variant,gpu", [("bad", 0), ("scratch", -1)])
def test_smoke_rejects_bad_identifiers(variant, gpu):
    with pytest.raises(ValueError):
        smoke_command(repo=ROOT, variant=variant, gpu=gpu, output="unused")


def test_reference_export_is_not_overwritten_or_silently_reused(tmp_path):
    source, split_config, genes = [tmp_path / name for name in ("source", "split", "genes")]
    for path in (source, split_config, genes):
        path.write_bytes(b"fixture")
    reference = tmp_path / "reference"
    inputs = dict(source=source, split_config=split_config, genes=genes)
    assert check_references(reference, **inputs) is False
    reference.mkdir()
    for name in ("validation.h5ad", "real.h5ad", "controls.h5ad"):
        (reference / name).write_bytes(b"fixture")
    with pytest.raises(RuntimeError, match="Incomplete"):
        check_references(reference, **inputs)
    manifest = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "split_config_sha256": sha256_file(split_config),
        "selected_genes_sha256": sha256_file(genes),
        "evaluation_genes": 2000,
    }
    (reference / "manifest.json").write_text(json.dumps(manifest))
    assert check_references(reference, **inputs) is True
    split_config.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="provenance"):
        check_references(reference, **inputs)
    assert (reference / "real.h5ad").read_bytes() == b"fixture"


def test_existing_upstream_is_not_reset_to_match_expected_revision(tmp_path, monkeypatch):
    (tmp_path / "external/PerturbDiff").mkdir(parents=True)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "wrong-commit\n")
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
    with pytest.raises(RuntimeError, match="not overwritten"):
        ensure_upstream(repo=tmp_path)
    assert commands == []


def test_installer_syntax_and_shared_pins():
    subprocess.run(["bash", "-n", ROOT / "scripts/server/setup_replogle_autodl.sh"], check=True)
    common = (ROOT / "environments/replogle-common.txt").read_text()
    cuda = (ROOT / "environments/replogle-cuda124.txt").read_text()
    macos = (ROOT / "environments/replogle-macos.txt").read_text()
    assert "cell-eval==0.6.6" in common
    assert "-r replogle-common.txt" in cuda and "-r replogle-common.txt" in macos
    assert "torch==2.5.1+cu124" in cuda
