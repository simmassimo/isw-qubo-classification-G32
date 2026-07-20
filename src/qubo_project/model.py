from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

    # Save classification statistics
    json_stats = {
        "classifier": clf.__class__.__name__,
        "n_samples": x_test.shape[0],
        "target_1_count": int(np.sum(y_test)),
        "target_1_percentage": np.mean(y_test) * 100,
        "class_0": {
            "precision": precision_score(y_test, y_pred, pos_label=0),
            "recall": recall_score(y_test, y_pred, pos_label=0),
            "f1_score": f1_score(y_test, y_pred, pos_label=0),
            "support": int(np.sum(y_test == 0)),
        },
        "class_1": {
            "precision": precision_score(y_test, y_pred, pos_label=1),
            "recall": recall_score(y_test, y_pred, pos_label=1),
            "f1_score": f1_score(y_test, y_pred, pos_label=1),
            "support": int(np.sum(y_test == 1)),
        },
        "roc_auc": roc_auc_score(y_test, y_pred),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
    }

    with open(classif_stats_json, 'w') as f:
        json.dump(json_stats, f, indent=4)

    with open(predictions_csv, 'w') as f:
        f.write(','.join(map(str, x_headers)) + ',prediction\n')
        for i in range(len(y_pred)):
            f.write(','.join(map(str, x_test[i])) + f',{y_pred[i]}\n')