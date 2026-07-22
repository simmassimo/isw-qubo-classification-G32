from qubo_project.preprocessing import fit_normalize
from qubo_project.preprocessing import write_report
from utils import *
import pytest

# input: BAD alfanumeric char in data row 1 
# assert: warning 'ignoring bad row' is made


fname = get_filenames('tests/data/', __file__)

test_data = fit_normalize(
    fname['input'],
    'target',
    fname['normalize'],
    fname['report'],
    0.05 )
    
#print(test_data)
write_report( fname['test_data'], test_data )

#report = get_report( fname['report'] )
    
assert find_warning(test_data['warnings'], 'ignoring bad row') 