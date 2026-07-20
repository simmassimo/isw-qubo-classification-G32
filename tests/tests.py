from ../src/qubo_project/preprocessing import fit_normalize
#from simple import fit_normalize
import json
import pytest

# main
# test to assert that output is numerical columns only
# method: 
#   read file that contains specifically engineered errors:
#       see file test01.csv 
#       that contains non numerical values and blank values and missing columns 
#   check the errors are present in the report.json
#       bad_rows parameter is not empty 
#       warning parameter contains certain messages

test_data = fit_normalize(
    'input.csv',         
#    'test01.csv',         
    'target',
    'normalise.csv',
    'report01.json',
    0.05 )
print(test_data)
  
def find_warning(arr, s) :
    for msg in arr:
            if s in msg: return True
    return False
  
# load/read json file
with open('report01.json') as file:
    report = json.load(file)
    file.close();

  
#assert 2==1   
#assert len(test_data['bad_rows']) < 0
assert find_warning(test_data['warnings'], 'ignoring bad row') # False when data is good and preprossesing.py is good too


# not finished yet 


