import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from baselines.adapter_gears import GEARSAdapter
from celldiffa.benchmark.gears_replogle import materialize_training_anndata
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit


def _write_split(path):
    values = {
        "pert_col": "gene",
        "control_pert": "non-targeting",
        "cell_line_key": "cell_line",
        "perturbseq_batch_col": "gem_group",
        "holdout_celltype": ["hepg2"],
        "holdout_pert": {"validation": ["A"], "test": ["B"]},
    }
    path.write_text(yaml.safe_dump(values), encoding="utf-8")


def _source(tmp_path):
    obs = pd.DataFrame(
        {
            "gene": ["non-targeting", "non-targeting", "A", "A", "B", "B", "C"],
            "cell_line": ["k562", "hepg2", "k562", "hepg2", "k562", "hepg2", "hepg2"],
            "gem_group": ["b1"] * 7,
        },
        index=[f"cell{i}" for i in range(7)],
    )
    adata = ad.AnnData(
        X=np.zeros((7, 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["unused"]),
    )
    adata.obsm["X_hvg"] = np.arange(14, dtype=np.float32).reshape(7, 2)
    path = tmp_path / "source.h5ad"
    adata.write_h5ad(path)
    return path


def test_perturbdiff_masks_and_real_test_validation(tmp_path):
    split_path = tmp_path / "split.yaml"
    _write_split(split_path)
    split = PerturbDiffSplit.from_yaml(split_path)
    source = ad.read_h5ad(_source(tmp_path))
    masks = split.masks(source.obs)
    np.testing.assert_array_equal(np.flatnonzero(masks["validation"]), [3])
    np.testing.assert_array_equal(np.flatnonzero(masks["test"]), [5])
    np.testing.assert_array_equal(np.flatnonzero(masks["train"]), [0, 1, 2, 4, 6])

    real = ad.AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"gene": ["non-targeting", "B"], "cell_line": ["hepg2", "hepg2"]}
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    split.validate_real_test(real)


def test_materialization_never_reads_validation_or_test_rows(tmp_path):
    split_path = tmp_path / "split.yaml"
    _write_split(split_path)
    split = PerturbDiffSplit.from_yaml(split_path)
    source = _source(tmp_path)

    pooled, counts = materialize_training_anndata(
        source,
        split=split,
        selected_genes=["g1", "g2"],
        mode="pooled",
        chunk_size=2,
    )
    np.testing.assert_array_equal(pooled.X.toarray(), np.arange(14).reshape(7, 2)[[0, 1, 2, 4, 6]])
    assert set(pooled.obs["condition"]) == {"ctrl", "A+ctrl", "B+ctrl", "C+ctrl"}
    assert set(pooled.obs["cell_type"]) == {"pooled"}
    assert counts["excluded_validation_rows"] == 1
    assert counts["excluded_test_rows"] == 1

    heldout, _ = materialize_training_anndata(
        source,
        split=split,
        selected_genes=["g1", "g2"],
        mode="heldout_only",
        chunk_size=3,
    )
    np.testing.assert_array_equal(heldout.X.toarray(), np.arange(14).reshape(7, 2)[[1, 6]])
    assert set(heldout.obs["condition"]) == {"ctrl", "C+ctrl"}


def test_gears_all_train_strategy_never_requests_simulation_split(tmp_path, monkeypatch):
    class FakePertData:
        def __init__(self, data_path, default_pert_graph=True):
            self.data_path = data_path
            self.default_pert_graph = default_pert_graph
            self.dataset_path = str(tmp_path / "processed")
            (tmp_path / "processed").mkdir()

        def new_data_process(self, dataset_name, adata):
            self.dataset_name = dataset_name
            self.adata = adata
            self.pert_names = np.array(["A"])

        def prepare_split(self, **kwargs):
            self.prepared = kwargs

        def get_dataloader(self, **kwargs):
            self.loader_args = kwargs
            self.dataloader = {"train_loader": [], "val_loader": []}

    class FakeGEARS:
        def __init__(self, pert_data, device, weight_bias_track):
            self.pert_data = pert_data

        def model_initialize(self, hidden_size):
            self.hidden_size = hidden_size

        def train(self, epochs, lr):
            self.train_args = (epochs, lr)

    monkeypatch.setitem(
        sys.modules,
        "gears",
        SimpleNamespace(PertData=FakePertData, GEARS=FakeGEARS),
    )
    training = ad.AnnData(
        X=np.ones((2, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"condition": ["ctrl", "A+ctrl"], "cell_type": ["pooled", "pooled"]}
        ),
        var=pd.DataFrame({"gene_name": ["g1", "g2"]}, index=["g1", "g2"]),
    )
    adapter = GEARSAdapter(data_path=str(tmp_path), device="cpu")
    adapter.fit(training, split_strategy="all_train", dataset_name="replogle_test")

    assert adapter._pert_data.prepared["split"] == "custom"
    assert adapter._pert_data.split == "no_test"
    assert set(adapter._pert_data.set2conditions) == {"train", "val"}
    assert adapter.is_trained
