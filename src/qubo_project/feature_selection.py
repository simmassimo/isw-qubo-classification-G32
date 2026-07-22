def select_features(
 normalized_csv: str, # Input dataset name
 reducedTrain_csv: str, # Name of output training dataset with reduced feat.
 reducedTest_csv: str, # Name of output test dataset with reduced features
 output_ottim_csv: str, # Name of output optimization data varying alpha
 output_json: str, # Name of output statistics and data file
 target_column: str, # Column name of target
 percTest: float = 0.30, # % of test data with respect to the dataset size
 percSelected: float = 0.20, # percentage of features to select
 allowance: int = 1, # Allowance of features to select
 seed: int = 42, # Seed for random repeatibility
 alpha_computations: int = 100, # Max. n. of optimizations varying alpha
): 
    '''
    Compute all the correlation coeffiecents and place them
    - in the U matrix, or
    - the V vector
    Pragmatic strategy to avoid memory overload
    - read the file line by line
    - store only a subset of columns on each passage determined by batch-size
    - always store the target column
    Once the columns are stored for each column pair
    - compute the correlation coeff and place it into the matrix at Uij and Uji
    - except for the col target correlation that goes into the V vector at Vi
    - use the standard Open source Scipy numpy routine to actual compute the correlations
    '''
    
    
    return Exception("select_features function is not implemented yet.")
