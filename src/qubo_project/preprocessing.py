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

def ReadCSV(input_csv: str):
    return Exception("ReadCSV function is not implemented yet.")

def SeparateTarget(csv, target_column: str):
    return Exception("SeparateTarget function is not implemented yet.")

def RemoveNullOrLowVarianceColumns(params, minPercValid: float):    
    return Exception("RemoveNullOrLowVarianceColumns function is not implemented yet.")

def NormalizeColumns(params):
    return Exception("NormalizeColumns function is not implemented yet.")