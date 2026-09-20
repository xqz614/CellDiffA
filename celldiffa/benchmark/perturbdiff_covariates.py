"""Keep dataset-subset category IDs consistent with released embedding tables."""

COVARIATE_VOCABULARIES = (
    ("pert_dict", "num_pert"),
    ("cell_type_dict", "num_celltype"),
    ("batch_dict", "num_batch"),
)


def align_checkpoint_covariates(cfg, datamodule, checkpoint_cfg):
    """Use checkpoint IDs, not subset-local IDs, before ``setup_dataset``.

    Keep the full saved vocabulary, including its unused entries, so indices
    still name the same trained embedding rows. Missing names are errors, never
    zero/unknown fallbacks. Validate everything before mutating any dictionary.
    """
    stages = ["train", *datamodule.all_split_names]
    if any(hasattr(datamodule, f"{stage}_dataset") for stage in stages):
        raise ValueError("Align checkpoint covariates before building datasets.")
    vocabularies, report = {}, {"version": "checkpoint_category_ids_v1"}
    for dictionary, count in COVARIATE_VOCABULARIES:
        saved = dict(checkpoint_cfg.get(dictionary) or {})
        current = dict(getattr(datamodule, dictionary))
        ids = list(saved.values())
        if (
            not saved
            or any(type(i) is not int for i in ids)
            or sorted(ids) != list(range(len(saved)))
        ):
            raise ValueError(f"Checkpoint {dictionary} must have unique contiguous integer IDs.")
        if checkpoint_cfg.get(count) != len(saved):
            raise ValueError(f"Checkpoint {count} does not match {dictionary}.")
        missing = sorted(set(current) - set(saved))
        if missing:
            raise ValueError(
                f"Checkpoint {dictionary} is missing runtime categories: {missing[:20]}"
            )
        vocabularies[dictionary] = saved
        report[dictionary] = {
            "runtime_categories": len(current),
            "checkpoint_categories": len(saved),
            "remapped_categories": sum(current[name] != saved[name] for name in current),
        }
    for dictionary, count in COVARIATE_VOCABULARIES:
        saved = vocabularies[dictionary]
        setattr(datamodule, dictionary, saved.copy())
        cfg.cov_encoding[dictionary] = saved.copy()
        cfg.cov_encoding[count] = len(saved)
    datamodule.checkpoint_covariate_alignment = report
    return report


def validate_checkpoint_covariates(datamodule, checkpoint_cfg):
    """Fail before sampling/resuming if any active label indexes a wrong row."""
    for dictionary, _ in COVARIATE_VOCABULARIES:
        saved = checkpoint_cfg[dictionary]
        active = getattr(datamodule, dictionary)
        if any(name not in saved or saved[name] != index for name, index in active.items()):
            raise ValueError(f"Runtime {dictionary} does not match checkpoint category IDs.")
        for stage in ["train", *datamodule.all_split_names]:
            dataset = getattr(datamodule, f"{stage}_dataset", None)
            if dataset is not None and dict(getattr(dataset.meta_cache, dictionary)) != dict(
                active
            ):
                raise ValueError(
                    f"{stage} dataset has stale {dictionary}; rebuild it after alignment."
                )
