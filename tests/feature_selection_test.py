import pytest
from qubo_project.feature_selection import select_features
import pandas as pd
import os

@pytest.fixture
def mock_csv(tmp_path):
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "feature2": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
        "target":   [0,   1,   0,   1,   0,   1,   0,   1],
    })
    path = tmp_path / "mock.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_feature_selection_produces_binary_vector(mock_csv, tmp_path):
    output_csv = tmp_path / "selected_features.csv"
    v = select_features(mock_csv, "target", str(output_csv))

    # Check if the returned vector is binary
    assert len(v) > 0 and set(v).issubset({0, 1})

def test_feature_selection_must_be_roughly_20_percent(mock_csv, tmp_path):
    output_csv = tmp_path / "selected_features.csv"
    v = select_features(mock_csv, "target", str(output_csv))

    # Check if at least 20% of the 1s in the vector are present
    assert sum(v) >= 0.15 * len(v) and sum(v) <= 0.25 * len(v), "Selected features should be roughly 20% of the total features."