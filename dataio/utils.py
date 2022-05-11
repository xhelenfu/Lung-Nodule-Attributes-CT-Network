import os
import re
import numpy as np
import nibabel as nib

"""
Alphanumerically sort a list
"""
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

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
Get directories of samples in fold split
"""
def get_data_dirs_split(patient_subset_txt, img_data_dir):
    with open(patient_subset_txt, 'r') as f:
        patient_subset = f.read().splitlines()

    ct_dirs = [(img_data_dir + '/' + x) for x in patient_subset]
    print('Number of samples: %d' %(len(patient_subset)))
    assert(len(ct_dirs) == len(patient_subset))
    ct_dirs = sorted_alphanumeric(ct_dirs)
    return ct_dirs

"""
Get image dimensions
"""
def get_image_dims(ct_dirs):
    sample_path = ct_dirs[0] + '/' + 'ct_axial.nii'
    img_sample = nib.load(sample_path)
    img_sample = np.array(img_sample.dataobj, dtype=np.float32)
    in_height = img_sample.shape[0]
    in_width = img_sample.shape[1]
    return in_height, in_width