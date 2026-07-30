import pytest
from qubo_project.feature_selection import select_features
import pandas as pd
import os

@pytest.fixture
def mock_csv(tmp_path):
    df = pd.DataFrame({
        "target":   [0,   1,   0,   1,   0,   1,   0,   1,  0,   1],
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature2": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.45],
    })
    path = tmp_path / "mock.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_feature_selection_produces_binary_vector(mock_csv, tmp_path):
    reduced_train_csv = str(tmp_path / "reduced_train.csv")
    reduced_test_csv = str(tmp_path / "reduced_test.csv")
    output_ottim_csv = str(tmp_path / "test_ottim.csv")
    output_json = str(tmp_path / "test.json")
    v = select_features(mock_csv, reduced_train_csv, reduced_test_csv, output_ottim_csv, output_json, "target")

    # Check if the returned vector is binary
    assert len(v) > 0 and set(v).issubset({0, 1})

def test_feature_selection_must_be_roughly_20_percent(mock_csv, tmp_path):
    reduced_train_csv = str(tmp_path / "reduced_train.csv")
    reduced_test_csv = str(tmp_path / "reduced_test.csv")
    output_ottim_csv = str(tmp_path / "test_ottim.csv")
    output_json = str(tmp_path / "test.json")

    perc = 0.2
    allowance = 1
    v = select_features(mock_csv, reduced_train_csv, reduced_test_csv, output_ottim_csv, output_json, "target", percSelected = perc, allowance = allowance)

    print(sum(v) / len(v))  # Print the proportion of selected features for debugging
    # result should be roughly 20% of the total features +- allowance
    assert (sum(v) / len(v)) >= 0.15 and (sum(v) / len(v)) <= 0.25, "Selected features should be roughly 20% of the total features."