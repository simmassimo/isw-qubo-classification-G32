from sklearn.ensemble import RandomForestClassifier
import numpy as np
from preprocessing import ReadCSV

def train(
 classifier: str, # type of classifier to use
 reducedTrain_csv: str, # training dataset
 target_column: str, # target column name
 model_path: str, # saved trained classifier
 metrics_json: str, # file with training statistics
 seed: int = 42,
):
    # open the csv file and read the data
    csv,_,_ = ReadCSV(reducedTrain_csv)

    if classifier == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise ValueError("Unsupported classifier type")

    clf.fit(csv[1:, :-1].astype(float), csv[1:, -1].astype(float))
    

def predict(
 reduced_Test_csv: str, # Input test set
 target_column: str, # Target column name
 model_path: str, # saved trained classifier to use
 predictions_csv: str, # Output predictions
 classif_stats_json: str, # File with classification stats
):
    return Exception("predict function is not implemented yet.")