from qubo_project.preprocessing import fit_normalize
from qubo_project.preprocessing import write_report
from utils import *
import pytest

# input: BAD column standard deviation
# assert: warnings: 'stdev too close to zero, so eliminating col: '


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
    
assert find_warning(test_data['warnings'], 'stdev too close to zero, so eliminating col: ') 