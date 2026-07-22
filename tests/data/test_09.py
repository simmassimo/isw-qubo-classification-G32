from qubo_project.preprocessing import fit_normalize
from qubo_project.preprocessing import write_report
from utils import *
import pytest
import os.path


# input: target column must contains only 0 or 1
# assert: 'target_column contains bad data'

fname = get_filenames('tests/data/', __file__)
    

test_data = fit_normalize(
    fname['input'],
    'target',
    fname['normalize'],
    fname['report'],
    0.05 )
      
#print(test_data)
test_data['normalized_ok'] = os.path.isfile(fname['normalize'])
write_report( fname['test_data'], test_data )

#report = get_report( fname['report'] )
    
assert 'target_column contains bad data at row:' in test_data['error']