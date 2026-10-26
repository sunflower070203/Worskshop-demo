import pandas as pd

from shift_audit import prepare_shift_data, summarize_missingness


def test_prepare_shift_data_excludes_id_and_target():
    train = pd.DataFrame(
        {"id": [10, 11], "f1": [1.0, None], "f2": [2.0, 3.0], "satellite_health": [1, 0]}
    )
    test = pd.DataFrame({"id": [20], "f1": [4.0], "f2": [None]})

    features, domain = prepare_shift_data(train, test, ["f1", "f2"])

    assert list(features.columns) == ["f1", "f2"]
    assert domain.tolist() == [0, 0, 1]
    assert pd.isna(features.iloc[1]["f1"])


def test_missingness_summary_uses_each_splits_own_denominator():
    train = pd.DataFrame({"f1": [1.0, None], "f2": [2.0, 3.0]})
    test = pd.DataFrame({"f1": [4.0], "f2": [None]})

    summary = summarize_missingness(train, test, ["f1", "f2"])

    assert summary["f1"] == {"train_rate": 0.5, "test_rate": 0.0, "difference_test_minus_train": -0.5}
    assert summary["f2"] == {"train_rate": 0.0, "test_rate": 1.0, "difference_test_minus_train": 1.0}
