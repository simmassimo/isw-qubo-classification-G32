import json
from pathlib import Path
  
def find_warning(arr, s) :
    for msg in arr:
            if s in msg: return True
    return False
  
# read a row form normalized file
#def get_normalize(file_name, row_nb):

# load/read json file
def get_report(file_name):
    with open(file_name) as file:
        report = json.load(file)
        file.close();
        return report

def get_filenames(dir_, file_py) :
    base_name = Path(file_py).stem
    test_nb = base_name.split('_')[1]
    files = {};
    files['input']  = dir_ + 'input'  + test_nb + '.csv'
    files['report'] = dir_ + 'report' + test_nb + '.json'
    files['normalize'] = dir_ + 'normalize' + test_nb + '.csv'
    files['test_data'] = dir_ + 'test_data' + test_nb + '.json'
    return files
