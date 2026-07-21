import re
import math
import json


def write_report( jsonfile, report ) :
    try:
        file = open(jsonfile, 'w')
    except OSError:
        print( 'CATASTROPHIC ERROR cannot open/write outInitalRes_json: ' + jsonfile)
        exit(99)
    json.dump(report, file ) #,indent=4)
    file.close();

def warning_error( test_data, msg: str ) :
    test_data['warnings'].append(msg)

def update_dropped_feature_names( report, arr_heads, arr_zeros ) :
    for i, value in enumerate(arr_heads):
        if arr_zeros[i] == 1 :
            report['dropped_feature_names'].append(arr_heads[i])

def update_n_kept_features( report, arr_zeros ) :
    count = 0
    for value in arr_zeros:
        if value == 0 : count += 1
    report['n_kept_features'] = count 

def zero_arr( arr, n ) :
    for i in range(n):  arr.append(0)

def write_arr( file, arr ) :
    out_line = ','.join(arr) + '\n'
    file.write( out_line )

def fit_normalize(
    input_csv: str = 'input.csv',           # Input dataset name
    target_column: str = 'target',          # column name of target - allows target to head any column it likes
    normalized_csv: str = 'normalise.csv',  # Name of output normalized data set
    outInitalRes_json: str = 'report.json', # Name of output statistics and data file
    minPercValid: float = 0.05,             # Minimum % of valid non-zero data for a column
):
    
    report =     {
        "n_input_features": 0,  # For now
        "n_kept_features": 0,   # These will also include id & target cols
        "dataset_size": 0,      # nb rows of input is this total_row_count
        "dataset_input_time": 0.00,      # secs
        "dataset_processing_time": 0.00, # secs
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
    
    # check for invalid arguments
    # Don't check for empty filename - should be caught by defaults
        
    if minPercValid < 0.0 or 1.0 < minPercValid :
        msg = 'minPercValid: ' + str(minPercValid) + ' not in range [0,1]'
        print( msg )
        test_data['error'] = msg
        write_report( outInitalRes_json, report )
        return test_data

    THRESHOLD = 1e-10
    COLS = 0 # later calculated from header row
    ID = 'id'

    # arrays of length determined by header (was 147)
    arr_sum = [] # reused for mean  
    arr_sum_sqr = [] # reused for sd  
    arr_zeros = [] # count, then 1/0 boolean
    arr_heads = [] # string labels

    # keep these indices global
    target_index = 0;
    id_index = 0;



    try:
        file_in = open(input_csv, 'r')
    except OSError:      
        msg = 'Could not open/read file: ' + input_csv
        print( msg )
        test_data['error'] = msg
        write_report( outInitalRes_json, report )
        return test_data
    try:
        file_out = open(normalized_csv, 'w')
    except OSError:
        file_in.close()
        msg = 'Could not open/write file: ' + normalized_csv
        print( msg )
        test_data['error'] = msg
        write_report( outInitalRes_json, report )
        return test_data

    with file_in:
        row_index = 0
        for line in file_in :      # FIRST PASS - get num. of COLS
        
            if row_index == 0: # HEADER
                # check header has no blanks etc
                pat_header_item = '[ ]*[^,]+[ ]*'
                pat_header_row = '^(' + pat_header_item + ',){2,}' + pat_header_item +'$'
                pattern = re.compile( pat_header_row );
                res = re.match(pattern, line) 
                if res == None:
                    msg = 'header has bad format'
                    print( msg )
                    test_data['error'] = msg
                    write_report( outInitalRes_json, report )
                    return test_data
                
                arr_heads = line.strip().split(',')
                COLS = len(arr_heads) 
                report['n_input_features'] = COLS
                if COLS < 3:
                    msg = 'bad header: got ' + str(COLS) + ' columns should be 3 or more'
                    print( msg )
                    test_data['error'] = msg
                    write_report( outInitalRes_json, report )
                    return test_data
                    
                # assign target_index using target_column name
                if not target_column in arr_heads : 
                    msg = 'bad header: no field found labelled ' + target_column
                    print( msg )
                    test_data['error'] = msg
                    write_report( outInitalRes_json, report )
                    return test_data
                target_index = arr_heads.index( target_column );
                
                # assign id_index
                if not 'id' in arr_heads : 
                    msg = 'bad header: no field found labelled ' + 'id' 
                    print( msg )
                    test_data['error'] = msg
                    write_report( outInitalRes_json, report )
                    return test_data
                id_index = arr_heads.index( 'id' );
                
                # first initial the arrays to 0
                zero_arr( arr_sum,     COLS )
                zero_arr( arr_sum_sqr, COLS )
                zero_arr( arr_zeros,   COLS )
                zero_arr( test_data['means'],COLS )
                zero_arr( test_data['sdevs'],COLS )
                
                row_index += 1
                    
            else: # DATA
                # pattern - not yet allowing 1.2e-5 engineering notation
                pat_numeric_item = '[ ]*-?[0-9]+(\.[0-9]+)?[ ]*'
                pat_numeric_row = '^(' + pat_numeric_item + ',){' + str(COLS-1) + '}' + pat_numeric_item +'$'
                pattern = re.compile( pat_numeric_row );
                
                # skip rows that fail regexp - reporting row_index to stderr
                res = re.match(pattern, line) 
                if res == None:
                    warning_error( test_data, 'ignoring bad row: ' + str(row_index))
                    test_data['bad_rows'].append(row_index)
                else:
                    # can now assume valid number of columns and valid numeric items
                    arr_vals = line.strip().split(',')  
                    test_data['valid_row_count'] += 1
                    for i, value in enumerate(arr_vals):
                        val = float(value)
                        arr_sum[i]     += val;
                        arr_sum_sqr[i] += (val*val);
                        if abs(val) < THRESHOLD: arr_zeros[i] += 1
                row_index += 1

        report['dataset_size'] = row_index -1 # omitting header_row
        if test_data['valid_row_count'] < 2:  
            msg = 'not enough valid numeric rows: ' + str(test_data['valid_row_count']) 
            print( msg )
            test_data['error'] = msg
            write_report( outInitalRes_json, report )
            return test_data

        # use arr_zeros to mark columns to exclude
        limit = (1.0 - minPercValid)*test_data['valid_row_count']    # e.g 95%
        for i, val in enumerate(arr_zeros):
            if   limit <= val : arr_zeros[i] = 1 # too many zeros
            else:               arr_zeros[i] = 0 # okay

        arr_zeros[target_index] = 0 # don't exclude target column


        # calc stats of sample
        # mean = sum / n
        # sd^2 = (sum_sqr - sum*sum/n)/(n-1)
        # sd = sqrt (sd^2)
        for i in range(COLS):
            if arr_zeros[i] == 0  and i != target_index and i != id_index:
                mean = arr_sum[i] / test_data['valid_row_count']                                          # sample mean
                sd   = math.sqrt( (arr_sum_sqr[i] - mean*arr_sum[i]) / (test_data['valid_row_count']-1) ) # sample sd
                if sd < THRESHOLD :  # then bad column, so re--classify column as almost-zero
                    warning_error( test_data, 'stdev too close to zero, so eliminating col: ' + arr_heads[i])
                    arr_zeros[i] = 1
                else:
                    test_data['means'][i] = mean
                    test_data['sdevs'][i] = sd
                   
        update_dropped_feature_names( report, arr_heads, arr_zeros )
        update_n_kept_features( report, arr_zeros )
        
        file_in.seek(0) # REWIND the file in order iterate the lines again.
        row_index = 0
        for line in file_in :      # SECOND PASS
        
            if row_index == 0: # HEADER
                # arr_heads = line.split(',')
                arr_out = []
                for i, value in enumerate(arr_heads) :
                    if arr_zeros[i] == 0: arr_out.append(value) #copy
                write_arr( file_out, arr_out )
                row_index += 1
                    
            else: # DATA
                arr_out = []
                if not row_index in test_data['bad_rows'] : # skip bad rows
                    arr_vals = line.strip().split(',')  
                    for i, value in enumerate(arr_vals) :
                        if i == target_index or i == id_index:
                            arr_out.append(value) # simple copy
                        else:
                            if arr_zeros[i] == 0 : 
                                # normalise 
                                # 1 subtract mean
                                # 2 divide by sd
                                val = float(value)
                                val = (val - test_data['means'][i]) / test_data['sdevs'][i]
                                arr_out.append(str(val))
                    write_arr( file_out, arr_out )
                row_index += 1

    # dump json here
    write_report( outInitalRes_json, report ) 
    # close all files here
    file_in.close();
    file_out.close();
    
    return test_data
