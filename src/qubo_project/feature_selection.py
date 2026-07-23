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
    THREE STAGES
    
    - 1 PRE-SORTING columns
        Create a new version of normalize.csv where each column has been sorted numerically
        Pragmatic strategy to avoid memory overload
        do the above for a subset of columns at a time
        re-reading the file for the subsequent subsets and appending the new cols
        Call the output sorted.csv
        include the target column in the sorting
        omit the 'id' field from the sorted.csv, completely
        
     - 2 ACCUMULATE STATS from sorted.csv in a single pass line by line
         Maintain the following arrays incrementally for each column 
         Maintain the following arrays incrementally for each column 
          arr_sum_u
          arr_sum_sqr_u 
          sum_uv - a pair-wise array uv to store the product of col_u and col_v
          to generate the col-index pairs use: 
          
            from itertools import combinations
            arr = range(0,COLS)
            for i, j in combinations(arr, 2):
                print(i, j)

          
      - 3 Construct the U matrix from the stats
          The U matrix Uij is computed from the numerator and denominator using the stable-sum formula
            num = n * sum_uv - sum_u * sum_v
            denom_term1 = n * sum_u2 - sum_u * sum_u
            denom_term2 = n * sum_v2 - sum_v * sum_v
            if denom_term1 <= 0 or denom_term2 <= 0:
            raise ValueError("correlation undefined for zero variance input")
            den = math.sqrt(de
        
      - 4 RUN QUBO on U matrix and V
      
    '''
    
    
    
    return Exception("select_features function is not implemented yet.")
