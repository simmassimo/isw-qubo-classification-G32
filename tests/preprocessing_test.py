import pytest
import pandas as pd
import os
from qubo_project.preprocessing import *

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


def test_read_csv_returns_correct_shape(mock_csv):
    csv = ReadCSV(mock_csv)
    assert csv.shape[0] == 8  # 8 rows of data
    assert csv.shape[1] == 3  # 2 features + 1 target column

def test_read_csv_with_missing_file_raises_error(tmp_path):
    missing_file_path = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        ReadCSV(str(missing_file_path))

def test_read_csv_with_empty_file_raises_error(tmp_path):
    empty_file_path = tmp_path / "empty.csv"
    empty_file_path.touch()  # Create an empty file
    with pytest.raises(ValueError):
        ReadCSV(str(empty_file_path))

def test_read_csv_with_malformed_csv_raises_error(tmp_path):
    malformed_csv_path = tmp_path / "malformed.csv"
    with open(malformed_csv_path, 'w') as f:
        f.write("feature1,feature2,target\n")
        f.write("1.0,0.1,0\n")
        f.write("2.0,0.9\n")  # Missing target value
        f.write("3.0,0.2,0\n")
    
    with pytest.raises(ValueError):
        ReadCSV(str(malformed_csv_path))

def test_read_csv_with_non_numeric_data_raises_error(tmp_path):
    non_numeric_csv_path = tmp_path / "non_numeric.csv"
    with open(non_numeric_csv_path, 'w') as f:
        f.write("feature1,feature2,target\n")
        f.write("1.0,0.1,0\n")
        f.write("2.0,abc,1\n")  # Non-numeric value in feature2
        f.write("3.0,0.2,0\n")
    
    with pytest.raises(ValueError):
        ReadCSV(str(non_numeric_csv_path))

def test_separate_target_with_missing_column_raises_error(mock_csv):
    csv = ReadCSV(mock_csv)
    with pytest.raises(ValueError):
        SeparateTarget(csv, "non_existent_target")

def test_separate_target_returns_correct_shapes(mock_csv):
    csv = ReadCSV(mock_csv)
    x_data, y_data = SeparateTarget(csv, "target")
    assert x_data.shape[0] == 8  # 8 rows of data
    assert x_data.shape[1] == 2  # 2 features
    assert y_data.shape[0] == 8  # 8 rows of target data
    assert y_data.shape[1] == 1  # 1 target column

def test_separate_target_with_non_numeric_target_raises_error(tmp_path):
    non_numeric_target_csv_path = tmp_path / "non_numeric_target.csv"
    with open(non_numeric_target_csv_path, 'w') as f:
        f.write("feature1,feature2,target\n")
        f.write("1.0,0.1,0\n")
        f.write("2.0,0.9,abc\n")  # Non-numeric value in target
        f.write("3.0,0.2,0\n")
    
    csv = ReadCSV(str(non_numeric_target_csv_path))
    with pytest.raises(ValueError):
        SeparateTarget(csv, "target")

def test_normalize_columns_with_zero_variance():
    data = np.array([[1.0, 2.0, 3.0],
                     [1.0, 5.0, 6.0],
                     [1.0, 8.0, 9.0]])
    normalized_data = NormalizeColumns(data)
    # The first column has zero variance and should be all zeros after normalization
    assert np.all(normalized_data[:, 0] == 0)
    # The other columns should be normalized to have mean 0 and std 1
    assert np.isclose(np.mean(normalized_data[:, 1]), 0)
    assert np.isclose(np.std(normalized_data[:, 1]), 1)
    assert np.isclose(np.mean(normalized_data[:, 2]), 0)
    assert np.isclose(np.std(normalized_data[:, 2]), 1)

def test_remove_almost_zero_columns():
    data = np.array([
        ["feature1", "feature2", "feature3"],
        [1.0, 0.00001, 3.0],
        [1.0, 0.00002, 6.0],
        [1.0, 0.00003, 9.0]
    ])
    filtered_data, n_features, filtered_names = RemoveNullOrLowVarianceColumns(data, minPercValid=0.5, variance_threshold=1e-4)
    # The second column should be removed due to low variance
    assert filtered_data.shape[1] == 2