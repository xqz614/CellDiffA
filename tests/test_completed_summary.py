import numpy as np
import pandas as pd
import pytest

from celldiffa.benchmark.metrics import PAPER_METRIC_NAMES
from scripts.baselines.summarize_replogle_completed import check_metrics


def table():
    return pd.DataFrame(
        {"perturbation": ["A", "B"], **{key: [0.2, 0.4] for key in ["R2", *PAPER_METRIC_NAMES]}}
    )


def test_complete_summary_requires_all_conditions_and_metrics():
    result = check_metrics(table(), {"A", "B"})
    assert len(result) == 14
    assert result["PDS_cos"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="coverage"):
        check_metrics(table().iloc[:1], {"A", "B"})
    with pytest.raises(ValueError, match="14 metrics"):
        check_metrics(table().drop(columns="MSE"), {"A", "B"})


def test_nonfinite_and_duplicate_results_are_not_complete():
    values = table()
    values.loc[0, "MSE"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        check_metrics(values, {"A", "B"})
    values = table()
    values.loc[1, "perturbation"] = "A"
    with pytest.raises(ValueError, match="coverage"):
        check_metrics(values, {"A", "B"})
