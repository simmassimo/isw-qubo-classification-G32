from sklearn.ensemble import RandomForestClassifier
import numpy as np
from preprocessing import ReadCSV, SeparateTarget
import joblib
import time
import json

def PrepareData(csv_file: str, target_column: str):
    t = time.time()
    csv = ReadCSV(csv_file)
    tnow = time.time()
    x_data, y_data = SeparateTarget(csv, target_column)
    x_headers = x_data[0, :]  # Save header row
    x_data = x_data[1:, :].astype(float)  # Exclude header row and convert to float
    y_data = y_data[1:, :].astype(float)  # Exclude header row and convert to float
    return x_data, y_data, x_headers, tnow - t

def train(
 classifier: str, # type of classifier to use
 reducedTrain_csv: str, # training dataset
 target_column: str, # target column name
 model_path: str, # saved trained classifier
 metrics_json: str, # file with training statistics
 seed: int = 42,
):
    classifier = classifier.lower().strip()
    x_train, y_train, x_headers, t_in = PrepareData(reducedTrain_csv, target_column)
    if x_train is None or y_train is None:
        raise ValueError("Training data could not be prepared. Check the input CSV and target column.")
    if classifier in ["random_forest", "rf", "randomforest", "random forest"]:
        json_stats["classifier"] = "random_forest"
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise ValueError("Unsupported classifier type")
    t = time.time()
    clf.fit(x_train.astype(float), y_train.astype(float))
    t_fit = time.time() - t
    # Saving classifier using joblib
    joblib.dump(clf, f'{model_path}')
    json_stats = {
        "seed": seed,
        "model_path": model_path,
        "training_dataset": reducedTrain_csv,
        "n_samples": x_train.shape[0],
        "n_features": x_train.shape[1],
        "target_1_percentage": np.mean(y_train) * 100,
        "dataset_input_time": round(t_in, 2),
        "training_time": round(t_fit, 2),
    }
    with open(metrics_json, 'w') as f:
        json.dump(json_stats, f, indent=4)

def predict(
 reduced_Test_csv: str, # Input test set
 target_column: str, # Target column name
 model_path: str, # saved trained classifier to use
 predictions_csv: str, # Output predictions
 classif_stats_json: str, # File with classification stats
):
    x_test, y_test, x_headers, _ = PrepareData(reduced_Test_csv, target_column)

    # Load the trained classifier
    clf = joblib.load(f'{model_path}')
    if clf is None:
        raise ValueError("Classifier not found. Are you sure the model was saved correctly?")

    # Make predictions
    y_pred = clf.predict(x_test.astype(float))
