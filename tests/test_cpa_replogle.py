import anndata as ad
import numpy as np
import pandas as pd

from scripts.baselines.run_cpa_replogle import prediction_inputs


def test_cpa_prediction_inputs_contain_only_observed_controls():
    controls = ad.AnnData(
        X=np.array([[1, 2], [3, 4]], dtype=np.float32),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    targets = pd.DataFrame({"gene": ["A", "B", "A"], "cell_line": ["hepg2"] * 3})
    query = prediction_inputs(controls, targets, seed=42)
    assert query.shape == (3, 2)
    assert query.obs.gene.tolist() == ["A", "B", "A"]
    assert all(any(np.array_equal(row, ctrl) for ctrl in controls.X) for row in query.X)
    assert query.var_names.equals(controls.var_names)
