import pytest
from qubo_project.model import train, predict, PrepareData
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

def test_prepare_data_returns_correct_shapes(mock_csv):
    x_data, y_data, x_headers, y_header, t_in = PrepareData(mock_csv, "target")
    
    assert x_data.shape[0] == 8  # 8 rows of data
    assert x_data.shape[1] == 2  # 2 features
    assert y_data.shape[0] == 8  # 8 rows of target data
    assert len(x_headers) == 2   # 2 feature headers
    assert y_header == "target"   # Target header is correct

def test_train_invalid_classifier_raises_error(mock_csv, tmp_path):
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    with pytest.raises(ValueError, match="Unsupported classifier type"):
        train("unsupported_classifier", mock_csv, "target", str(model_path), str(metrics_path), seed=42)

def test_train_with_invalid_target_column_raises_error(mock_csv, tmp_path):
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    with pytest.raises(ValueError, match="Target column 'invalid_target' not found in CSV headers."):
        train("random_forest", mock_csv, "invalid_target", str(model_path), str(metrics_path), seed=42)

def test_model_file_is_created(tmp_path, mock_csv):
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    train("random_forest", mock_csv, "target", str(model_path), str(metrics_path), seed=42)

    assert model_path.exists()


def test_predictions_file_is_created(tmp_path, mock_csv):
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.csv"
    classif_stats_path = tmp_path / "classif_stats.json"

    # Train the model
    train("random_forest", mock_csv, "target", str(model_path), str(metrics_path), seed=42)

    # Make predictions
    predict(mock_csv, "target", str(model_path), str(predictions_path), str(classif_stats_path))

    assert predictions_path.exists()