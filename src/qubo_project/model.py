def train(
 classifier: str, # classifier to use
 reducedTrain_csv: str, # training dataset
 target_column: str, # target column name
 model_path: str, # saved trained classifier
 metrics_json: str, # file with training statistics
 seed: int = 42,
):
    return Exception("train function is not implemented yet.")

def predict(
 reduced_Test_csv: str, # Input test set
 target_column: str, # Target column name
 model_path: str, # saved trained classifier to use
 predictions_csv: str, # Output predictions
 classif_stats_json: str, # File with classification stats
):
    return Exception("predict function is not implemented yet.")