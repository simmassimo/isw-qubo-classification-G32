from qubo_project.preprocessing import fit_normalize
from qubo_project.preprocessing import write_report
from utils import *
import pytest

# input: BAD field in header
# assert: fatal error 'header has bad format'


fname = get_filenames('tests/data/', __file__)

test_data = fit_normalize(
    fname['input'],
    'target',
    fname['normalize'],
    fname['report'],
    0.05 )
    
print(test_data)
write_report( fname['test_data'], test_data )

report = get_report( fname['report'] )
    
assert test_data['error'] == 'bad header: no field found labelled target'