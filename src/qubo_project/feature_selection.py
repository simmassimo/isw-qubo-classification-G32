import re
import math
import json
import time
import numpy as np
from pathlib import Path
from itertools import combinations

'''
    We assume that the input file normalize_csv has not been
    modified since it was written by pre_processing.py
    Thus we omit here all that format checking done in pre_processing.py
    Thus there will be much less occasion to to apply pytest.
    
    
    THREE STAGES
    
    - 1 PRE-RANKING columns
        Apply numerical ranking follows.
        PRAGMATIC STRATEGY to avoid memory overload
        Do this for a subset of columns at a time
        re-reading the file for the subsequent subsets.
        Exclude the 'id' column from any subset.
        Exclude the 'target' column from any subset.
        From each column create its ranking as a vector
        Write out each subset the ranked vectors as individual lines
        in a file ranking_XXX.csv 
        - where XXX is the column index of the first column of the subset.
        For the the 'target' column create its ranking as a vector
        and write it out to target.csv
        include the target column in the sorting
        
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
      
'''


# Globals so to keep error reporting code tidy
report_filename = ''
report =     {
    "n_features": 95,
    "target_ratio": 0.20,
    "target_k": 19,
    "allowance": 1,
    "n_selected": 19,
    "alpha": 0.344,
    "selected_vector": [],
    "selected_feature_names": [],
    "algorithm": "simulated_annealing",
    "seed": 42,
    "alpha_computations": 6,
    "percTest": 0.30,
    "training_dataset_size": 14000,
    "test_dataset_size": 6000,
    "q_matrix_creation_time": 2.53,
    "mean_optimization_time": 0.23,
    "std_dev_optimization_time": 0.044,
}
    
test_data = {    
    "valid_row_count":  0,
    "bad_rows": [],         # row indices
    "zeros": [],            # indexed by kept column
    "means": [],            # ...
    "sdevs": [],            # ...
    "warnings": []#,
    #"error": ''
}

def write_report() :
    try:
        file = open(report_filename, 'w')
    except OSError:
        print( 'CATASTROPHIC ERROR cannot open/write outInitalRes_json: ' + jsonfile)
        return
    json.dump(report, file )
    file.close();

def fatal_error( msg: str ) :
    print( msg )
    test_data['error'] = msg
    write_report()

def write_arr( file, arr ) :
    out_line = ','.join(arr) + '\n'
    file.write( out_line )

def bad_header_regex( line ) :
    pat_header_item = '[ ]*[^,]+[ ]*'
    pat_header_row = '^(' + pat_header_item + ',){2,}' + pat_header_item +'$'
    pattern = re.compile( pat_header_row );
    
    return None == re.match(pattern, line)

###############################################################################################
def process_header ( line, test_data, target_column ) : # returns test_data

    # check header has no blanks etc
    if bad_header_regex( line ):
        test_data['error'] = 'header has bad format'
        return 
    
    # create array of column header labels
    test_data['header'] = line.strip().split(',')
    
    # assign target_index using target_column name
    if not target_column in test_data['header'] : 
        test_data['error'] = 'bad header: no field found labelled ' + target_column
        return 
    test_data['target_index'] = test_data['header'].index( target_column );
    
    
    return 

###############################
def rank_and_write_subset_to_file( file_in, test_data, cache_dir, col_offset, subset_size): 
    
    subset_cols = [[] for _ in range(subset_size)]
    
    # REWIND the file & skip the header
    file_in.seek(0) 
    line = file_in.readline() #header

    for line in file_in :
        row = line.strip().split(',')

        for i in range(subset_size) :
            indx = i + col_offset
            subset_cols[i].append(row[indx])
            #np.append(subset_cols[i], row[indx], axis=None)

    # open file to store subset_cols - one row per col!
    rank_filename = cache_dir + '/rank_' + str(col_offset).zfill(2) + '.csv' # e.g. 'rank_01.csv'
    try:
        file_rank = open(rank_filename, 'w')
    except OSError:
        test_data['error'] = 'Could not open/read file: ' + rank_filename 
        return test_data
            
    # do RANKING then write to file one by one
    for i in range(subset_size) :
        rank_arr = np.array(subset_cols[i]).argsort().argsort().tolist() # RANKING numpy magic
        str_arr = (str(x) for x in rank_arr)
        write_arr( file_rank, str_arr )
        
    file_rank.close()
    return test_data
        

###############################################################################################
def select_features(
 normalized_csv: str, # Input dataset name
 #reducedTrain_csv: str, # Name of output training dataset with reduced feat.
 #reducedTest_csv: str,  # Name of output test dataset with reduced features
 output_ottim_csv: str, # Name of output optimization data varying alpha
 output_json: str, # Name of output statistics and data file
 target_column: str, # Column name of target
 percTest: float = 0.30, # % of test data with respect to the dataset size
 percSelected: float = 0.20, # percentage of features to select
 allowance: int = 1, # Allowance of features to select
 seed: int = 42, # Seed for random repeatibility
 alpha_computations: int = 100, # Max. n. of optimizations varying alpha
):
       
    # assign a global copy for convenience   
    global report_filename
    report_filename = output_json
        
    # check for invalid argument
    if percTest < 0.0 or 1.0 < percTest :
        fatal_error('percTest: ' + str(percTest) + ' not in range [0,1]')
        return test_data

    # check for invalid argument
    if percSelected < 0.0 or 1.0 < percSelected :
        fatal_error('percTest: ' + str(percSelected) + ' not in range [0,1]')
        return test_data

    # check for invalid argument
    if len(target_column) < 1:
        fatal_error('target_column: parameter is blank')
        return test_data

    try:
        file_in = open(normalized_csv, 'r') # input this time
    except OSError:      
        fatal_error('Could not open/read file: ' + normalized_csv )
        return test_data
        
    try:  # lock the file, so any error comes now, not after much processing
        file_out = open(output_ottim_csv, 'w')
    except OSError:
        file_in.close()
        fatal_error('Could not open/write file: ' + normalized_csv )
        return test_data

    with file_in:

        # start TIMER
        t_start = time.time()
        
        # READ HEADER
        line = file_in.readline()
        process_header ( line, test_data, target_column )
        if hasattr(test_data, 'error'):
            fatal_error(test_data['error'])
            return test_data

        COLS = len(test_data['header'])

        # Setup to ignore 'id' column
        if( 'id' in test_data['header'] ) :
            COLS = COLS - 1
            expected_target_index = 1
        else:
            expected_target_index = 0
        
        if COLS < 3:
            test_data['error'] = 'bad header: got ' + str(COLS) + ' columns should be 3 or more'
            return 

        if( test_data['target_index'] != expected_target_index ) :
            fatal_error('target column not in expected col : ' + str(expected_target_index) )
            return test_data
       
        report['n_features'] = COLS # includes target but not id field

        # must MKDIR cache if it does not exist
        cache_dir = 'cache'
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        SUBSET_SIZE = 5
        nb_subsets = COLS // SUBSET_SIZE # integer division
        remainder  = COLS % SUBSET_SIZE    
        target_index = test_data['target_index']
        
        print('COLS :' + str(COLS))
        print('SUBSET_SIZE :' + str(SUBSET_SIZE))
        print('nb_subsets :' + str(nb_subsets))
        print('remainder :' + str(remainder))
        print('target_index :' + str(target_index))
        
        for i in range(nb_subsets) :
            offset = target_index + i*SUBSET_SIZE
            rank_and_write_subset_to_file( file_in, test_data, cache_dir, offset, SUBSET_SIZE ) 
            if hasattr(test_data, 'error'):
                fatal_error(test_data['error'])
                return test_data
        
        if 0 < remainder :
            offset = target_index + nb_subsets*SUBSET_SIZE
            rank_and_write_subset_to_file( file_in, test_data, cache_dir, offset, remainder )
            if hasattr(test_data, 'error'):
                fatal_error(test_data['error'])
                return test_data
         
    write_report() 
    # close all open files here
    file_in.close();
    file_out.close();
    
    return test_data

    
print (select_features('normalise.csv', 'output_ottim_csv', 'report2.json','target'))
