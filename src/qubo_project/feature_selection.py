import re
import math
import json
import time
import numpy as np
import openjij as oj
from pathlib import Path
from itertools import combinations


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
def rank_subset_and_store( array_2d, file_in, offset_in, offset_out, subset_size): 
    
    subset_cols = [[] for _ in range(subset_size)]
    
    # REWIND the file & skip the header
    file_in.seek(0) 
    line = file_in.readline() #header

    for line in file_in :
        row = line.strip().split(',')

        for i in range(subset_size) :
            indx = i + offset_in
            subset_cols[i].append(row[indx])
           
    # do RANKING then store in array_2d
    for i in range(subset_size) :
        rank_arr = np.array(subset_cols[i]).argsort().argsort().astype(np.uint32) # RANKING with numpy
        array_2d[offset_out+i] = rank_arr; # this only works because rank_arr is np & the right length

########################
def new_vector(u) : # to be stored in indexed array
    sum_u  = sum(u)
    sum_u2 = sum(x*x for x in u)
    return {'sum': sum_u,'sum_sqr': sum_u2, 'data': u}
    
def pearson_corr_numpy(i, j, U, stats):
    """
    U[i], U[j]: the two arrays
    returns their Pearson correlation coefficient 
    (float)
    stats help avoid recomputing means, etc
    """
    u = np.asarray(U[i], dtype=float)
    v = np.asarray(U[j], dtype=float)

    if not stats[i] : # u is {} so compute mean and cen2
        stats[i]['mean'] = u.mean()
        u_center         = u - stats[i]['mean'] # array
        stats[i]['cen2'] = np.sum(u_center**2)
    else:
        u_center = u - stats[i]['mean'] # array
          
    if not stats[j] : # v is {} so compute mean and cen2
        stats[j]['mean'] = v.mean()
        v_center         = v - stats[j]['mean'] # array
        stats[j]['cen2'] = np.sum(v_center**2)
    else:
        v_center = v - stats[i]['mean'] # array
        
    num = np.sum(u_center * v_center)
    den = np.sqrt( stats[i]['cen2'] * stats[j]['cen2'] )
    if den == 0:
        raise ValueError("correlation undefined for zero variance input")
    return abs(float(num / den))

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
 
        # run thru file once to get other array dimension
        nb_ROWS=0
        for line in file_in :
            nb_ROWS += 1;

        
        # In memory RANKING array each cell is unsigned 32 bit
        # ~900MB may be needed for 1.5 milion lines of 100 cols
        # array rows are the ranked columns data (of length nb_ROWS)
        # there will be COLS number of these rows 
        
        array_2d = np.empty( (COLS,nb_ROWS), dtype=np.uint32)

        # we will read the file many times reading only subset of columns at a time
        SUBSET_SIZE = 4
        nb_subsets = COLS // SUBSET_SIZE # integer division
        remainder  = COLS % SUBSET_SIZE    
        target_index = test_data['target_index']
        
        print('COLS :' + str(COLS))
        print('nb_ROWS :' + str(nb_ROWS))
        print('SUBSET_SIZE :' + str(SUBSET_SIZE))
        print('nb_subsets :' + str(nb_subsets))
        print('remainder :' + str(remainder))
        print('target_index :' + str(target_index))
        
   
        for i in range(nb_subsets) :
            offset_out  = i*SUBSET_SIZE
            offset_in   = target_index + offset_out
            rank_subset_and_store( array_2d, file_in, offset_in, offset_out, SUBSET_SIZE ) 

        if 0 < remainder :
            offset_out  = nb_subsets*SUBSET_SIZE
            offset_in   = target_index + offset_out
            rank_subset_and_store( array_2d, file_in, offset_in, offset_out, remainder ) 

        # done reading normalize_csv so can close file
        file_in.close();
        
        # print('array_2d :' + str( print(array_2d[0:5, 0:5]) ))

        # Create QUBO matrix to be filled with rho value 
        Q_triu = np.zeros( (COLS-1,COLS-1), dtype=np.float64)
        Q_diag = np.zeros( (COLS-1), dtype=np.float64)
        stats = [{}] * COLS # to fill with {'mean','cen2'}
        
        # generate array of pairs of column indices (i,j)
        # - without the (j,i) or the (i,i)
        pairs = combinations(range(0,COLS), 2)
        for i, j in pairs :
            rho = pearson_corr_numpy(i, j, array_2d, stats)

            if i == 0 : # the ranked target vector
                Q_diag[j-1] =  rho # a vector
            else:
                Q_triu[i-1][j-1] = rho # a matrix
            
        # create and print Q_matrix for a variety of alpha values

        for i in range(1,10):
            alpha = i/10.0
            # print( str(i) + ' alpha : ' + str(alpha) + ' Q:' )
            Q_matrix = (1-alpha) * Q_triu.copy()
            # replace the diagonal with Q_diag * alpha
            np.fill_diagonal(Q_matrix, alpha * Q_diag)   
            # print(Q_matrix[0:5, 0:5])
            
            # Create a solver
            sampler = oj.SASampler()

            # Solve the QUBO problem
            response = sampler.sample_qubo(Q_matrix)

            # Get the best solution
            X_vector = response.first.sample
            
            count = np.sum(np.array(list(X_vector.values())) == 1)
            
            print("alpha: ", alpha, "count: ", count)
            # print("X_vector: ", X_vector)

    
#print (select_features('normalise.csv', 'output_ottim_csv', 'report2.json','target'))
select_features('normalise.csv', 'output_ottim_csv', 'report2.json','target')
