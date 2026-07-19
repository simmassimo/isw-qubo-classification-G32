from sklearn.ensemble import RandomForestClassifier
import numpy as np
from preprocessing import ReadCSV, SeparateTarget
import joblib

def PrepareData(csv_file: str, target_column: str):
    csv = ReadCSV(csv_file)
    x_data, y_data = SeparateTarget(csv, target_column)
    x_headers = x_data[0, :]  # Save header row
    x_data = x_data[1:, :].astype(float)  # Exclude header row and convert to float
    y_data = y_data[1:, :].astype(float)  # Exclude header row and convert to float
    return x_data, y_data, x_headers

def train(
 classifier: str, # type of classifier to use
 reducedTrain_csv: str, # training dataset
 target_column: str, # target column name
 model_path: str, # saved trained classifier
 metrics_json: str, # file with training statistics
 seed: int = 42,
):
    classifier = classifier.lower().strip()
    x_train, y_train, x_headers = PrepareData(reducedTrain_csv, target_column)
    if classifier in ["random_forest", "rf", "randomforest", "random forest"]:
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise ValueError("Unsupported classifier type")

    clf.fit(x_train.astype(float), y_train.astype(float))
    # Saving classifier using joblib
    joblib.dump(clf, f'{model_path}/model_{classifier}.pkl')

def predict(
 reduced_Test_csv: str, # Input test set
 target_column: str, # Target column name
 model_path: str, # saved trained classifier to use
 predictions_csv: str, # Output predictions
 classif_stats_json: str, # File with classification stats
):
    x_test, y_test, x_headers = PrepareData(reduced_Test_csv, target_column)

