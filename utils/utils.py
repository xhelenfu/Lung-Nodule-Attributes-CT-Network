import os
import datetime as dt
import json
import collections
import re

"""
Alphanumerically sort a list
"""
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
    
"""
Make directory if doesn't exist
"""
def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

"""
Get all files in a directory with a specific extension
"""
def get_files_list(path, ext_array=['.tif']):
    files_list = list()
    dirs_list = list()

    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            if any(x in file for x in ext_array):
                files_list.append(os.path.join(root, file))
                folder = os.path.dirname(os.path.join(root, file))
                if folder not in dirs_list:
                    dirs_list.append(folder)

    return files_list, dirs_list
    
"""
Read json config file
"""
def json_file_to_pyobj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json2obj(open(filename).read())

"""
Get the ID of current experiment
"""
def get_experiment_id(make_new, load_dir, fold_id):
    if make_new is False:
        if load_dir == 'last':
            folders = next(os.walk('experiments'))[1]
            folders = [x for x in folders if ('fold' + str(fold_id) + '_') in x]
            folders = sorted_alphanumeric(folders)
            folder_last = folders[-1]
            timestamp = folder_last.replace('\\','/')
        else:
            timestamp = load_dir
    else:
        timestamp = 'fold' + str(fold_id) + '_' + dt.datetime.now().strftime("%Y_%B_%d_%H_%M_%S")
    
    return timestamp
