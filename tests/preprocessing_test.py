import os
import pytest

from qubo_project.preprocessing import fit_normalize
from utils import find_warning, get_filenames

DATA_DIR = 'tests/data/'


@pytest.fixture
def fname(request):
    # request.param is the dataset number, e.g. '01' -> input01.csv / normalize01.csv / ...
    return get_filenames(DATA_DIR, request.param)


# input: file not there
# assert: fatal error 'Could not open/read file'
@pytest.mark.parametrize('fname', ['00'], indirect=True)
def test_fit_normalize_with_missing_file_raises_error(fname):
    with pytest.raises(FileNotFoundError, match="Could not open/read file"):
        fit_normalize(
            fname['input'],
            'target',
            fname['normalize'],
            fname['report'],
            0.05
        )


# input: BAD alfanumeric char in data row 1
# assert: warning 'ignoring bad row' is made
@pytest.mark.parametrize('fname', ['01'], indirect=True)
def test_fit_normalize_with_bad_row_raises_warning(fname):
    test_data = fit_normalize(
        fname['input'],
        'target',
        fname['normalize'],
        fname['report'],
        0.05 )

    assert find_warning(test_data['warnings'], 'ignoring bad row')


# input: EMPTY field in data row 1
# assert: warning 'ignoring bad row' is made
@pytest.mark.parametrize('fname', ['02'], indirect=True)
def test_fit_normalize_with_empty_field_raises_warning(fname):
    test_data = fit_normalize(
        fname['input'],
        'target',
        fname['normalize'],
        fname['report'],
        0.05 )

    assert find_warning(test_data['warnings'], 'ignoring bad row')


# input: BAD field in header
# assert: fatal error 'header has bad format'
@pytest.mark.parametrize('fname', ['03'], indirect=True)
def test_fit_normalize_with_bad_header_raises_error(fname):
    with pytest.raises(ValueError, match="bad header: no field found labelled target"):
        fit_normalize(
            fname['input'],
            'target',
            fname['normalize'],
            fname['report'],
            0.05 )


# input: EMPTY field in header
# assert: fatal error 'header has bad format'
@pytest.mark.parametrize('fname', ['04'], indirect=True)
def test_fit_normalize_with_empty_header_raises_error(fname):
    with pytest.raises(ValueError, match="header has bad format"):
        fit_normalize(
            fname['input'],
            'target',
            fname['normalize'],
            fname['report'],
            0.05 )


# input: data set has less than 3 columns (assuming no id column)
# assert: fatal error 'header has bad format'
@pytest.mark.parametrize('fname', ['05'], indirect=True)
def test_fit_normalize_with_too_few_columns_raises_error(fname):
    with pytest.raises(ValueError, match="header has bad format"):
        fit_normalize(
            fname['input'],
            'target',
            fname['normalize'],
            fname['report'],
            0.05
        )


# input: Too few data rows
# assert: fatal error 'not enough valid numeric rows: '
@pytest.mark.parametrize('fname', ['06'], indirect=True)
def test_fit_normalize_with_too_few_rows_raises_error(fname):
    with pytest.raises(ValueError, match="not enough valid numeric rows: "):
        fit_normalize(
            fname['input'],
            'target',
            fname['normalize'],
            fname['report'],
            0.05 )


# input: BAD column standard deviation
# assert: warnings: 'stdev too close to zero, so eliminating col: '
@pytest.mark.parametrize('fname', ['07'], indirect=True)
def test_fit_normalize_with_bad_column_stdev_raises_warning(fname):
    test_data = fit_normalize(
        fname['input'],
        'target',
        fname['normalize'],
        fname['report'],
        0.05 )

    assert find_warning(test_data['warnings'], 'stdev too close to zero, so eliminating col: ')


# input: normalized.csv not created
# assert: normalize file exists after fit_normalize runs
@pytest.mark.parametrize('fname', ['08'], indirect=True)
def test_fit_normalize_creates_normalized_file(fname):
    fit_normalize(
        fname['input'],
        'target',
        fname['normalize'],
        fname['report'],
        0.05 )

    assert os.path.exists(fname['normalize'])


# input: target column must contain only 0 or 1
# assert: 'target_column contains bad data'
@pytest.mark.parametrize('fname', ['09'], indirect=True)
def test_fit_normalize_with_bad_target_column_format_raises_error(fname):
    with pytest.raises(ValueError, match="target_column contains bad value"):
        fit_normalize(
            fname['input'],
            'target',
            fname['normalize'],
            fname['report'],
            0.05 )
