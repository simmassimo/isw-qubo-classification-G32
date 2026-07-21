import pytest
from qubo_project.model import train, predict
import pandas as pd
import os

@pytest.fixture
def sample_data():
    df = pd.read_csv("../data/trial_dataset_ISW.csv")
    sample = df.sample(n=20, random_state=42)  # random_state makes it reproducible
    sample.to_csv("../data/sample.csv", index=False)


def test_model_file_is_created():
    #remove file if exists
    if os.path.exists("../data/trial_model.pkl"):
        os.remove("../data/trial_model.pkl")
    #grab a random column from the dataset to use as target
    target_column = df.columns[-1].strip()  # Select the last column as target

    train(
        "random_forest",
        "../data/sample.csv",
        target_column,
        "../data/trial_model.pkl",
        "../data/trial_metrics.json")
    assert os.path.exists("../data/trial_model.pkl"), "Model file was not created."

def test_predictions_file_is_created():
    #remove file if exists
    if os.path.exists("../data/trial_predictions.csv"):
        os.remove("../data/trial_predictions.csv")
    #train the model first if necessary
    if not os.path.exists("../data/trial_model.pkl"):
        train(
            "random_forest",
            "../data/sample.csv",
            target_column,
            "../data/trial_model.pkl",
            "../data/trial_metrics.json")
    predict(
        "../data/sample.csv",
        "target",
        "../data/trial_model.pkl",
        "../data/trial_predictions.csv",
        "../data/trial_classif_stats.json")
    assert os.path.exists("../data/trial_predictions.csv"), "Predictions file was not created."