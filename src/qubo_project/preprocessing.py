import numpy as np
import time

def ReadCSV(input_csv: str):
    try:
        with open(input_csv, 'r') as f:
            csv = [line.strip().split(',') for line in f.readlines()]
        csv = np.array(csv)
        size = csv.shape[0]
        n_ftrs_raw = csv.shape[1] - 1  # Exclude target column
        return csv, size, n_ftrs_raw
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return [], 0, 0

def SeparateTarget(csv, target_column: str):
    headers = csv[0]
    target_index = np.where(headers == target_column)[0][0]
    if target_index == -1:
        raise ValueError(f"Target column '{target_column}' not found in CSV headers.")
    target = csv[:, target_index]
    #remove the target column from the params including header
    params = np.delete(csv, target_index, axis=1)
    return params, target
    
    

def AlmostZero(x, tol=1e-4):
    return abs(x) < tol

def NonAlmostZero(x, tol=1e-4):
    return not AlmostZero(x, tol)

def CountNonAlmostZero(arr, tol=1e-4):
    return sum(1 for x in arr if NonAlmostZero(x, tol))

def CountAlmostZero(arr, tol=1e-4):
    return sum(1 for x in arr if AlmostZero(x, tol))

def RemoveNullOrLowVarianceColumns(params, minPercValid: float, variance_threshold: float): 
    params_h = params[0]  # header row
    params_v = params[1:].astype(float)   
    filtered_params_names = []
    delete_mask = np.zeros(params_v.shape[1], dtype=bool)
    for icol, col in enumerate(params_v.T):
        delete_mask[icol] = CountAlmostZero(col, tol=variance_threshold) / len(col) >= minPercValid
    params_filtered = params[:, ~delete_mask]
    filtered_params_names = params_h[delete_mask]
    return params_filtered, params_filtered.shape[1] - 1, filtered_params_names


def NormalizeColumns(params):
    params_h = params[0]
    params_v = params[1:].astype(float)
    #implement z-score normalization
    normalized_v = (params_v - np.mean(params_v, axis=0)) / np.std(params_v, axis=0)
    return np.vstack([params_h, normalized_v])

def WriteCSV(normalized_csv: str, params, target):
    try:
        with open(normalized_csv, 'w') as f:
            for row in params:
                f.write(','.join(map(str, row)) + '\n')
            f.write(','.join(map(str, target)) + '\n')
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def fit_normalize(
 input_csv: str, # Input dataset name
 target_column: str, # column name of target
 normalized_csv: str, # Name of output normalized data set
 outInitalRes_json: str, # Name of output statistics and data file
 minPercValid: float = 0.05, # Minimum % of valid non-zero data for a column
 variance_threshold: float = 1e-4, # Minimum variance threshold for a column
):
    json_stats = {}
    t = time.time()
    csv, size, n_ftrs_raw = ReadCSV(input_csv)
    tnow = time.time()
    json_stats["dataset_input_time"] = round(tnow - t,2)
    json_stats["dataset_size"] = size
    print(f"Read CSV with {size} rows and {n_ftrs_raw} features")
    t = time.time()
    params,target = SeparateTarget(csv,target_column)
    json_stats["n_input_features"] = n_ftrs_raw
    params, n_ftrs_clean, rmvd_ftrs = RemoveNullOrLowVarianceColumns(params,minPercValid, variance_threshold)
    print(f"Removed {n_ftrs_raw - n_ftrs_clean} features with low variance or too many near-zero values")
    json_stats["n_kept_features"] = n_ftrs_clean
    json_stats["dropped_features_names"] = rmvd_ftrs
    params = NormalizeColumns(params)
    tnow = time.time()
    json_stats["dataset_processing_time"] = round(tnow - t,2)
    WriteCSV(normalized_csv, params, target)
    return params, target, json_stats


p,t,s = fit_normalize(
    input_csv="data/trial_dataset_ISW.csv",
    target_column="target",
    normalized_csv="data/trial_dataset_ISW_normalized.csv",
    outInitalRes_json="data/trial_dataset_ISW_stats.json",
    minPercValid=0.05,
    variance_threshold=1e-6)

print(s)