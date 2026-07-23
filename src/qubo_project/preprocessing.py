import re
import math
import json
import time

# Globals so to keep error reporting code tidy
report_filename = ''
report =     {
    "n_input_features": 0,  # For now
    "n_kept_features": 0,   # These will also include id & target cols
    "dataset_size": 0,      # nb rows of input is this total_row_count
    "dataset_input_time": 0.0,      # secs
    "dataset_processing_time": 0.0, # secs
    "dropped_feature_names": [] # e.g. ["feature_1", "feature_20"]
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

ALMOST_ZERO = 1e-10 
# end of globals

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

def update_dropped_feature_names( report, arr_heads, arr_zeros ) :
    for i, value in enumerate(arr_heads):
        if arr_zeros[i] == 1 :
            report['dropped_feature_names'].append(arr_heads[i])

def update_n_kept_features( report, arr_zeros ) :
    count = 0
    for value in arr_zeros:
        if value == 0 : count += 1
    report['n_kept_features'] = count 

def write_arr( file, arr ) :
    out_line = ','.join(arr) + '\n'
    file.write( out_line )

def bad_header_regex( line ) :
    pat_header_item = '[ ]*[^,]+[ ]*'
    pat_header_row = '^(' + pat_header_item + ',){2,}' + pat_header_item +'$'
    pattern = re.compile( pat_header_row );
    return None == re.match(pattern, line)

def bad_row_regex( line, COLS ) : # pattern - not yet allowing for 1.2e-5 engineering notation
    pat_numeric_item = '[ ]*-?[0-9]+(\.[0-9]+)?[ ]*'
    pat_numeric_row = '^(' + pat_numeric_item + ',){' + str(COLS-1) + '}' + pat_numeric_item +'$'
    pattern = re.compile( pat_numeric_row );
    return None == re.match(pattern, line)


###############################################################################################

def fit_normalize(
    input_csv: str = 'input.csv',           # Input dataset name
    target_column: str = 'target',          # column name of target - allows target to head any column it likes
    normalized_csv: str = 'normalise.csv',  # Name of output normalized data set
    outInitalRes_json: str = 'report.json', # Name of output statistics and data file
    minPercValid: float = 0.05,             # Minimum % of valid non-zero data for a column
):
    
    # assign a global copy for convenience   
    global report_filename
    report_filename = outInitalRes_json
        
    # check for invalid arguments
    if minPercValid < 0.0 or 1.0 < minPercValid :
        fatal_error('minPercValid: ' + str(minPercValid) + ' not in range [0,1]')
        return test_data

    COLS = 0 # later calculated from header row
    ID = 'id'

    # keep these indices global
    target_index = 0;
    id_index = 0;


    try:
        file_in = open(input_csv, 'r')
    except OSError:      
        fatal_error('Could not open/read file: ' + input_csv )
        return test_data
    try:
        file_out = open(normalized_csv, 'w')
    except OSError:
        file_in.close()
        fatal_error('Could not open/write file: ' + normalized_csv )
        return test_data

    with file_in:

        # start TIMER
        t_start = time.time()

        line = file_in.readline() # read header on first line
        
        # check header has no blanks etc
        if bad_header_regex( line ):
            fatal_error('header has bad format')
            return test_data
        
        # create array of column header labels
        arr_heads = line.strip().split(',')
        
        COLS = len(arr_heads) # COLS is the total number of column found in the header
        report['n_input_features'] = COLS
        if COLS < 3:
            fatal_error('bad header: got ' + str(COLS) + ' columns should be 3 or more')
            return test_data
            
        # assign target_index using target_column name
        if not target_column in arr_heads : 
            fatal_error('bad header: no field found labelled ' + target_column )
            return test_data
        target_index = arr_heads.index( target_column );
        
        # assign id_index
        if not 'id' in arr_heads : 
            fatal_error('bad header: no field found labelled ' + 'id' )
            return test_data
        id_index = arr_heads.index( 'id' );
        
        # first initial the arrays to 0
        arr_sum     = [0] * COLS
        arr_sum_sqr = [0] * COLS
        arr_zeros   = [0] * COLS
        test_data['means'] = [0] * COLS 
        test_data['sdevs'] = [0] * COLS
        
        row_index = 1
                    
        for line in file_in :      # FIRST PASS of data rows
            
                # skip rows that fail regexp - reporting row_index to stderr
                if bad_row_regex( line, COLS ):
                    test_data['warnings'].append('ignoring bad row: ' + str(row_index))
                    test_data['bad_rows'].append(row_index)
                else:
                    # can now assume valid number of columns and valid numeric items
                    arr_vals = line.strip().split(',')
                    if (arr_vals[target_index] != '1') and (arr_vals[target_index] != '0'):
                        fatal_error('target_column contains bad data at row: ' + str(row_index) )
                        return test_data
                        
                    test_data['valid_row_count'] += 1
                    # compile the column stats sum & sum of sqrs & the count of almost-zero items
                    for i, value in enumerate(arr_vals):
                        val = float(value)
                        arr_sum[i]     += val;
                        arr_sum_sqr[i] += (val*val);
                        if abs(val) < ALMOST_ZERO: arr_zeros[i] += 1
                row_index += 1

        # stop TIMER
        t_end = time.time()
        report["dataset_input_time"] = round(t_end - t_start,2)
        
        report['dataset_size'] = row_index -1 # omitting header_row
        
        if test_data['valid_row_count'] < 2:  
            fatal_error('not enough valid numeric rows: ' + str(test_data['valid_row_count']) )
            return test_data


        # re-start TIMER
        t_start = time.time()


        # Use arr_zeros to mark columns to exclude
        limit = (1.0 - minPercValid)*test_data['valid_row_count']    # e.g 95%
        for i, val in enumerate(arr_zeros):
            if   limit <= val : arr_zeros[i] = 1 # too many zeros
            else:               arr_zeros[i] = 0 # okay

        arr_zeros[target_index] = 0 # don't exclude target column
        arr_zeros[id_index] = 0     # don't exclude id column


        # calc stats of sample
        # mean = sum / n
        # sd^2 = (sum_sqr - sum*sum/n)/(n-1)
        # sd = sqrt (sd^2)
        for i in range(COLS):
            if arr_zeros[i] == 0  and i != target_index and i != id_index:
                mean = arr_sum[i] / test_data['valid_row_count']                                          # sample mean
                sd   = math.sqrt( (arr_sum_sqr[i] - mean*arr_sum[i]) / (test_data['valid_row_count']-1) ) # sample sd
                if sd < ALMOST_ZERO :  # then bad column, so re--classify column as almost-zero
                    test_data['warnings'].append('stdev too close to zero, so eliminating col: ' + arr_heads[i])
                    arr_zeros[i] = 1
                else:
                    test_data['means'][i] = mean
                    test_data['sdevs'][i] = sd
                   
        update_dropped_feature_names( report, arr_heads, arr_zeros )
        update_n_kept_features( report, arr_zeros )
        
        # REWIND the file in order iterate all the lines again.
        file_in.seek(0) 
        line = file_in.readline() # skip header on first line

        # write out the original header omitting eliminated columns
        arr_out = []
        for i, value in enumerate(arr_heads) :
            if arr_zeros[i] == 0: arr_out.append(value) #copy
        write_arr( file_out, arr_out )

        row_index = 1
        for line in file_in :      # SECOND PASS of data rows
            arr_out = []
            if not row_index in test_data['bad_rows'] : # skip bad rows
                arr_vals = line.strip().split(',')  
                for i, value in enumerate(arr_vals) :
                    if i == target_index or i == id_index:
                        arr_out.append(value) # simple copy
                    else:
                        if arr_zeros[i] == 0 : 
                            # normalise: 1 subtract mean, 2 divide by sd
                            val = float(value)
                            val = (val - test_data['means'][i]) / test_data['sdevs'][i]
                            arr_out.append(str(val))
                write_arr( file_out, arr_out )
            row_index += 1
        
        # stop TIMER
        t_end = time.time()
        report["dataset_processing_time"] = round(t_end - t_start,2)

    write_report() 
    # close all open files here
    file_in.close();
    file_out.close();
    
    return test_data

print (fit_normalize('data/trial_dataset_ISW.csv'))
