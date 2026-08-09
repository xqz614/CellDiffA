from types import SimpleNamespace

from scripts.baselines.perturbdiff_sampling_entrypoint import patch_covariate_paths


def test_runtime_paths_replace_cluster_paths_without_mutating_checkpoint_config():
    checkpoint_cfg = {
        "celltype_encoding": "onehot",
        "celltype_embedding_path": "/projects/authors/celltype.pkl",
        "gene_embedding_path": ["/projects/authors/gene.pkl"],
        "pert_embedding_path": "/projects/authors/pert.pkl",
        "drug_embedding_path": "/projects/authors/drug.pkl",
        "replogle_gene_embedding_path": "/projects/authors/replogle.pkl",
        "hidden_dim": 128,
    }
    runtime_cfg = SimpleNamespace(
        celltype_encoding="llm",
        get=lambda key, default=None: {
            "celltype_embedding_path": "/data/celltype.pkl",
            "gene_embedding_path": ["/data/gene.pkl"],
            "pert_embedding_path": "/data/pert.pkl",
            "drug_embedding_path": "/data/drug.pkl",
            "replogle_gene_embedding_path": "/data/replogle.pkl",
        }.get(key, default),
    )

    patched = patch_covariate_paths(checkpoint_cfg, runtime_cfg)

    assert patched["celltype_encoding"] == "llm"
    assert patched["celltype_embedding_path"] == "/data/celltype.pkl"
    assert patched["gene_embedding_path"] == ["/data/gene.pkl"]
    assert patched["replogle_gene_embedding_path"] == "/data/replogle.pkl"
    assert patched["hidden_dim"] == 128
    assert checkpoint_cfg["replogle_gene_embedding_path"] == "/projects/authors/replogle.pkl"
