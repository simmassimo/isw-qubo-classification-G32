import json

def find_warning(arr, s) :
    for msg in arr:
            if s in msg: return True
    return False

# load/read json file
def get_report(file_name):
    with open(file_name) as file:
        report = json.load(file)
        file.close();
        return report

# build the dict of per-scenario file paths for a given dataset number, e.g. '01'
def get_filenames(dir_, test_nb) :
    files = {};
    files['input']  = dir_ + 'input'  + test_nb + '.csv'
    files['report'] = dir_ + 'report' + test_nb + '.json'
    files['normalize'] = dir_ + 'normalize' + test_nb + '.csv'
    files['test_data'] = dir_ + 'test_data' + test_nb + '.json'
    return files
