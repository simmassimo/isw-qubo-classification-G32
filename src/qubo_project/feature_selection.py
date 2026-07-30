import argparse
import re
import math
import json
import time
import numpy as np
import openjij as oj
from pathlib import Path

###############################
def write_report(output_json, report) :
    try:
        file = open(output_json, 'w')
    except OSError:
        print( 'CATASTROPHIC ERROR cannot open/write output_json: ' + output_json)

    json.dump(report, file )
    file.close();

###############################
def write_arr( file, arr ) :
    out_line = ','.join(arr) + '\n'
    file.write( out_line )

###############################
def bad_header_regex( line ) :
    pat_header_item = '[ ]*[^,]+[ ]*'
    pat_header_row = '^(' + pat_header_item + ',){2,}' + pat_header_item +'$'
    pattern = re.compile( pat_header_row );
    
    return None == re.match(pattern, line)

###############################################################################################
def process_header ( line, target_column, output_json, report ) :

    # check header has no blanks etc
    if bad_header_regex( line ):
        write_report(output_json, report)
        raise ValueError('header has bad format')
    
    # create array of column header labels
    header = line.strip().split(',')
    
    # assign target_index using target_column name
    if not target_column in header : 
        write_report(output_json, report)
        raise ValueError('bad header: no field found labelled ' + target_column )
        
    # check it is zero
    target_index = header.index( target_column );
    if( target_index != 0 ) :
        write_report(output_json, report)
        raise ValueError('target column not in expected col : 0')
        
    return header

###############################
def rank_subset_and_store( rankings, file_in, offset, subset_size): 
    
    subset_cols = [[] for _ in range(subset_size)]
    
    # REWIND the file & skip the header
    file_in.seek(0) 
    line = file_in.readline() #header

    for line in file_in :
        row = line.strip().split(',')

        for i in range(subset_size) :
            indx = i + offset
            subset_cols[i].append(row[indx])
           
    # do RANKING then store in rankings
    for i in range(subset_size) :
        rank_arr = np.array(subset_cols[i]).argsort().argsort().astype(np.uint32) # RANKING with numpy
        rankings[offset+i] = rank_arr; #  works because rank_arr is np & the correct length


#######################
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

########################################################
def get_mean_stdv(sum_, sum_sqr, n) :
    mean = sum_/n
    if 0 < (n-1.0) :
        stdv = math.sqrt(( sum_sqr - mean*sum_ ) / (n-1.0) )
    else:
        stdv  = 1.0
    return mean, stdv
    
#####################################       
def binary_search(Q_dict, report, file_out ) :
    """
    binary search
    Find alpha in [low, high] such that f(alpha) is closest to target_k.
    f(alpha) must be monotonically increasing.
    """
    # get parameter
    target_k = report['target_k']
    seed = report['seed']
    allowance = report['allowance']
    max_iter = report['alpha_computations']
    n_features = report['n_features']
    
    # initialise optimization params
    best_alpha = 0.0
    best_k = 0
    best_diff = n_features
    best_vec = []
    alpha = 0

    # initalise search range
    low   = 0.0
    high  = 1.0
    
    #initialise timing stats
    sum_secs = 0.0
    sum_sqr_secs = 0.0 
    nb_runs = 0
    file_out.write( 'alpha, time,  n_features, cost \n' ) # csv header

    for _ in range(max_iter):
        
        alpha = (low + high) / 2.0 # mid-point
        if nb_runs == 0 : alpha = 0.95
        cost, k, vec, secs = cost_function( alpha, Q_dict, seed )
        file_out.write( str(alpha) + ',' + str(secs) + ',' + str(k) + ',' + str(cost) + '\n' )
        diff = abs(k - target_k)
        #print( 'alfa, k, diff, best_diff: ',alpha, k, diff, best_diff )
        
        # increment stats
        sum_secs += secs
        sum_sqr_secs += (secs*secs)
        nb_runs += 1

        # Track best seen so far
        if diff < best_diff:
            best_alpha = alpha
            best_cost = cost
            best_k = k
            best_diff = diff
            best_vec = vec

        # If we hit the target within allowance, we can stop
        if diff <= allowance:
            break

        # Decide which half to keep
        if k < target_k: low = alpha
        else:           high = alpha
        
        '''
        # Stop if interval too small - non optimum solution
        if high - low < 0.1:
            print('break too small')
            break
        '''
        
    mean, stdv = get_mean_stdv(sum_secs,sum_sqr_secs, nb_runs)
    report["mean_optimization_time"] = mean
    report["std_dev_optimization_time"] = stdv
    report["alpha_computations"] = nb_runs
    return best_alpha, best_k, best_vec

########################################################
def cost_function( alpha, Q_dict, seed_ ):
    # start TIMER
    t_start = time.time() 

    Q_scaled = {}
    for (i, j), val in Q_dict.items():
        if i == j:
            Q_scaled[(i, i)] = alpha * val          # diagonal
        else:
            Q_scaled[(i, j)] = (1 - alpha) * val    # off-diagonal

    # Create a solver SA = Simulated Annealing
    sampler = oj.SASampler()
    
    # Solve the QUBO problem
    response = sampler.sample_qubo(Q_scaled, seed = seed_, sparse = False)
                                       
    # Rate the solution by computing count
    vec  = response.first.sample
    cost = response.first.energy
    vec = np.array(list(vec.values())) #un-dict-ification
    count = np.sum(vec == 1)
    
    # stop TIMER
    t_end = time.time()
    secs = round(t_end - t_start,2)

    return cost, count, vec, secs
    
###############################################################################################
def select_features(
 normalized_csv: str, # Input dataset name
 reducedTrain_csv: str, # Name of output training dataset with reduced feat.
 reducedTest_csv: str,  # Name of output test dataset with reduced features
 output_ottim_csv: str, # Name of output optimization data varying alpha
 output_json: str, # Name of output statistics and data file
 target_column: str, # Column name of target
 percTest: float = 0.30, # % of test data with respect to the dataset size
 percSelected: float = 0.20, # percentage of features to select
 allowance: int = 1, # Allowance of features to select
 seed: int = 42, # Seed for random repeatibility
 alpha_computations: int = 100, # Max. n. of optimizations varying alpha
):
    
    LAMBDA = 10.0 # the PENALTY used when constructing the Q_matrix

    report =     {
        "n_features": 95,
        "target_ratio": 0.20, # 20% same as percSelected
        "target_k": 19, # 20% of 95
        "allowance": 1, 
        "n_selected": 19,  # a result ~ "target_k"
        "alpha": 0.344,    # a result
        "selected_vector": [],  # a result (boolean)
        "selected_feature_names": [], # a result (strings)
        "algorithm": "simulated_annealing",
        "seed": 42,
        "alpha_computations": 6, # a result 
        "percTest": 0.30, # 30 %
        "training_dataset_size": 14000, # 70% of 20000
        "test_dataset_size": 6000,      # 30% of 20000
        "q_matrix_creation_time": 2.53,
        "mean_optimization_time": 0.23,
        "std_dev_optimization_time": 0.044,
    }
        
    # check for invalid argument
    if len(target_column) < 1:
        raise ValueError('target_column: parameter is blank')

    # check for invalid argument
    if percTest < 0.0 or 1.0 < percTest :
        raise ValueError('percTest: ' + str(percTest) + ' not in range [0,1]')
    report["percTest"] = percTest


    # check for invalid argument
    if percSelected < 0.0 or 1.0 < percSelected :
        raise ValueError('percSelected: ' + str(percSelected) + ' not in range [0,1]')
    report["target_ratio"] = percSelected

    # check for invalid argument
    if allowance < 0 :
        raise ValueError('allowance: ' + str(allowance) + ' is less than 0')
    report["allowance"] = allowance

    # check for invalid argument
    if seed < 0 :
        raise ValueError('seed: ' + str(seed) + ' is less than 0')
    report["seed"] = seed
        
    # check for invalid argument
    if alpha_computations < 1 :
        raise ValueError('alpha_computations: ' + str(alpha_computations) + ' is less than 1')
    report["alpha_computations"] = alpha_computations # the max to tried

    try:
        file_in = open(normalized_csv, 'r') # input in feature_selection
    except OSError:      
        raise FileNotFoundError('Could not open/read file: ' + normalized_csv )
        
    try:  # lock the file, so any error comes now, not after much processing
        file_train = open(reducedTrain_csv, 'w')
    except OSError:
        raise FileNotFoundError('Could not open/read file: ' + reducedTrain_csv )

    try:  # lock the file, so any error comes now, not after much processing
        file_test = open(reducedTest_csv, 'w')
    except OSError:
        raise FileNotFoundError('Could not open/read file: ' + reducedTest_csv )

    try:  # lock the file, so any error comes now, not after much processing
        file_out = open(output_ottim_csv, 'w')
    except OSError:
        raise FileNotFoundError('Could not open/read file: ' + output_ottim_csv )
        
    # check for invalid argument
    if len(output_json) < 1:
        raise ValueError('output_json: parameter is blank')


    with file_in:

        # start TIMER
        t_start = time.time() #q_matrix_creation_time
        
        # READ HEADER
        line = file_in.readline()
        header = process_header ( line, target_column, output_json, report )

        COLS = len(header)

        # 'id' column will NEVER be present - see preprocessing.py
        if COLS < 3:
            write_report(output_json, report)
            raise ValueError('bad header: got ' + str(COLS) + ' columns should be 3 or more')

        report['n_features'] = COLS # includes target
        target_k = int(COLS * percSelected)  # used later
        report['target_k']   = target_k

 
        # run thru file once to get other array dimension (yes I know!)
        nb_ROWS=0
        for line in file_in :
            nb_ROWS += 1;

        
        # In memory RANKING array each cell is unsigned 32 bit
        # ~900MB may be needed for 1.5 milion lines of 100 cols
        # array rows are the ranked columns data (of length nb_ROWS)
        # there will be COLS number of these rows 
        
        rankings = np.empty( (COLS,nb_ROWS), dtype=np.uint32)

        # we will read the file several times, reading only subset of columns at a time
        # which reduces memory needed to store columns
        SUBSET_SIZE = 4
        nb_subsets = COLS // SUBSET_SIZE # integer division
        remainder  = COLS % SUBSET_SIZE    
        
        '''
        # debugging
        print('COLS :' + str(COLS))
        print('nb_ROWS :' + str(nb_ROWS))
        print('SUBSET_SIZE :' + str(SUBSET_SIZE))
        print('nb_subsets :' + str(nb_subsets))
        print('remainder :' + str(remainder))
        '''
   
        for i in range(nb_subsets) :
            offset = i*SUBSET_SIZE
            rank_subset_and_store( rankings, file_in, offset, SUBSET_SIZE ) 

        if 0 < remainder :
            offset= nb_subsets*SUBSET_SIZE
            rank_subset_and_store( rankings, file_in, offset, remainder ) 
            
        # debugging
        # print('rankings :' + str( print(rankings[0:5, 0:5]) ))

        # Create QUBO matrix (as Q_dict) to be filled with rho value 
        K = target_k
        Q_dict = {}
        stats = [{}] * COLS # to fill with {'mean','cen2'} entries
        
        # LAMBDA=10 - declared at top of this function
        #diag case j==0
        for j in range(1,COLS):
           Q_dict[(j,j)] = pearson_corr_numpy(0, j, rankings, stats) + LAMBDA * (1 - 2*K) # linear term

           
        #triU
        for i in range(1,COLS):
            for j in range(i+1,COLS):
                Q_dict[(i,j)] = pearson_corr_numpy(i, j, rankings, stats) + 2 * LAMBDA # quadratic term
        
        # debugging
        #print('Q_dict:')
        #print(Q_dict)
        
        # stop TIMER
        t_end = time.time()
        report["q_matrix_creation_time"] = round(t_end - t_start,2)
        
        # optimise Q_matrix for a variety of alpha values
        
        X_vector = np.zeros(COLS - 1, dtype=bool)

        ###################################################
        # run binary search to find alpha best k and the vector
        
        alpha, best_k, best_vec = binary_search( Q_dict, report, file_out )

        # debugging
        # print( 'best alpha: ', alpha, 'best_k: ', best_k )
        
        X_vector = [x == 1 for x in best_vec]
        report["alpha"] = alpha
        report["n_selected"] = int(best_k)
        report["algorithm"] = "simulated_annealing"
        
        selected_cols = np.concatenate(([True], X_vector))
        reduced_header = [h for h, keep in zip(header, selected_cols) if keep]
        
        report["selected_vector"] = selected_cols.tolist()
        report["selected_feature_names"] = reduced_header
        report["algorithm"] = "simulated_annealing",

        ####################################################
       
        # Read normalize_csv line by line
        # split current line into fields (header included)
        # write out a copy that contain the field that are True in selected_cols
        
        test_n  = int(percTest*nb_ROWS)
        train_n = nb_ROWS - test_n
        
        report["training_dataset_size"] = train_n
        report["test_dataset_size"]     = test_n  
        
        file_in.seek(0)  # rewind normalize_csv to start
        line = file_in.readline() # skip header
        
        # write both headers
        write_arr( file_train, reduced_header )
        write_arr( file_test,  reduced_header )

        for i, line in enumerate(file_in) :
            row = line.strip().split(',')
            reduced_row = np.array(row)[selected_cols]  # a boolean filter
            if i < train_n:
                write_arr( file_train,  reduced_row )
            else:
                write_arr( file_test,   reduced_row )

    file_in.close()
    file_test.close()
    file_train.close()
    file_out.close()
    write_report(output_json, report)
    
    #print(selected_cols)
    return selected_cols
###########################################################

def main():
    parser = argparse.ArgumentParser(description='Feature Selection')
    
    # Positional arguments (required)
    # Optional argument 
    parser.add_argument('--in-normalized',        type=str, default='normalized.csv',
        help='Input normalized.csv file path')
    parser.add_argument('--out-train',            type=str, default='reducedTrain.csv',
        help='Output training_reduced.csv file path')
    parser.add_argument('--out-test',             type=str, default='reducedTest.csv',
        help='Output test_reduced.csv file path')
    parser.add_argument('--out-optimizations',    type=str, default='output_ottim_csv',
        help='Output optimizations.csv file path')
    parser.add_argument('--out-json',             type=str, default='report2.json',
        help='Output feature_selection_result.json file path')
    parser.add_argument('--target',               type=str, default='target',
        help='Name of target column in input csv file')
    
    parser.add_argument('--perc-selected', type=float, default='0.2',
                        help='Perc-selected expressed a decimal')
    parser.add_argument('--allowance', type=int, default='1',
                        help='Allowance expressed an integer')
    parser.add_argument('--perc-test', type=float, default='0.3',
                        help='Perc-test expressed a decimal')
    parser.add_argument('--seed', type=int, default='42',
                        help='seed expressed an integer')
    parser.add_argument('--alpha-computations', type=int, default='10',
                        help='alpha-computations (maxIter) expressed an integer')
 
    args = parser.parse_args()
 
    select_features(
         args.in_normalized,
         args.out_train,
         args.out_test,
         args.out_optimizations,
         args.out_json,
         args.target,
         args.perc_test,
         args.perc_selected,
         args.allowance,
         args.seed,
         args.alpha_computations
        )

if __name__ == '__main__':  # uncomment these line to get standard behaviour
    main()
main()
