import numpy as np

def ReadCSV(input_csv: str):
    with open(input_csv, 'r') as f:
        csv = [line.strip().split(',') for line in f.readlines()]
    return csv

def SeparateTarget(csv, target_column: str):
    return np.array([row for row in csv[1:]]), np.array([row[csv[0].index(target_column)] for row in csv[1:]])

def RemoveNullOrLowVarianceColumns(params, minPercValid: float):    
    # rotate the params to work with columns
    params_rotated = np.rot90(params)
    # Filter out columns with too many null or low variance values
    filtered_params = [col for col in params_rotated if np.count_nonzero(col) / len(col) >= minPercValid]
    # Rotate back
    return np.rot90(filtered_params, k=-1)

def NormalizeColumns(params):
    #implement z-score normalization
    params_r = np.rot90(params)
    for row in params_r:
        m = np.mean(row)
        s = np.std(row)
        if s != 0:
            row = [(x - m) / s for x in row]
        else:
            row = [0 for x in row]
    return np.rot90(params_r, k=-1)

def fit_normalize(
 input_csv: str, # Input dataset name
 target_column: str, # column name of target
 normalized_csv: str, # Name of output normalized data set
 outInitalRes_json: str, # Name of output statistics and data file
 minPercValid: float = 0.05, # Minimum % of valid non-zero data for a column
):
  csv = ReadCSV(input_csv)
  params,target = SeparateTarget(csv,target_column)
  params = RemoveNullOrLowVarianceColumns(params,minPercValid)
  params = NormalizeColumns(params)
  return params, target