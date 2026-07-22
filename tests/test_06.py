from qubo_project.preprocessing import fit_normalize
from qubo_project.preprocessing import write_report
from utils import *
import pytest

# input: Too few data rows
# assert: fatal error 'not enough valid numeric rows: '


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
    
assert 'not enough valid numeric rows: ' in test_data['error']