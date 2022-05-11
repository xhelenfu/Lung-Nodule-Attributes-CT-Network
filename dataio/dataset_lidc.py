import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import sys
import os
import imgaug.augmenters as iaa
import nibabel as nib

from .utils import *

class DatasetLIDC(data.Dataset):
    def __init__(self, data_sources, fold_id, fold_mean, fold_std, feature_ids, isTraining=True):

        # Check valid data directories
        if not os.path.exists(data_sources.img_data_dir):
            sys.exit("Invalid images directory %s" %data_sources.img_data_dir)
        if not os.path.exists(data_sources.feature_labels_file):
            sys.exit("Invalid feature labels path %s" %data_sources.feature_labels_file)

        self.img_data_dir = data_sources.img_data_dir
        self.fold_splits = data_sources.fold_splits
        self.feature_labels_file = data_sources.feature_labels_file
        self.fold_id = fold_id
        self.fold_mean = fold_mean
        self.fold_std = fold_std
        self.feature_ids = feature_ids
        self.isTraining = isTraining

        # Fold splits
        if isTraining:
            patient_subset_txt = self.fold_splits + '/' + str(self.fold_id) + '_train.txt'
        else:
            patient_subset_txt = self.fold_splits + '/' + str(self.fold_id) + '_test.txt'

        self.ct_dirs = get_data_dirs_split(patient_subset_txt, self.img_data_dir)

        # Determine the size of input images
        self.in_height, self.in_width = get_image_dims(self.ct_dirs)

        # All the labels for this data subset
        self.labels_df = pd.read_csv(self.feature_labels_file, index_col=0)

    # Training data augmentation
    def augment_data(self, batch_raw, n_slices):
        # Original, horizontal
        random_flip = np.random.randint(2, size=1)[0]
        # 0, 90, 180, 270
        random_rotate = np.random.randint(4, size=1)[0]
        # z-ordering of slices
        random_inv = np.random.randint(2, size=1)[0]

        # Flip
        if random_flip == 0:
            batch_flip = batch_raw
        else:
            batch_flip = iaa.Flipud(1.0)(images=batch_raw)
                
        # Rotate
        if random_rotate == 0:
            batch_rotate = batch_flip
        elif random_rotate == 1:
            batch_rotate = iaa.Rot90(1, keep_size=True)(images=batch_flip)
        elif random_rotate == 2:
            batch_rotate = iaa.Rot90(2, keep_size=True)(images=batch_flip)
        else:
            batch_rotate = iaa.Rot90(3, keep_size=True)(images=batch_flip)

        # Reverse z-ordering
        images_aug_array = np.zeros(batch_raw.shape, 'float32')
        for i in range(n_slices):
            if random_inv == 1:
                images_aug_array[n_slices-i-1,:,:] = np.squeeze(np.array(batch_rotate[i]))
            else:
                images_aug_array[i,:,:] = np.squeeze(np.array(batch_rotate[i]))
                
        return images_aug_array

    # Input normalisation
    def normalise_images(self, imgs):
        return (imgs - self.fold_mean)/self.fold_std

    # Load the slices of a nodule
    def load_slices_vol(self, files_path):
        vol = nib.load(files_path)
        vol = np.array(vol.dataobj, dtype=np.float32)
        vol = np.swapaxes(vol, 0, -1)
        return vol

    def __len__(self):
            return len(self.ct_dirs)

    def __getitem__(self, index):
            # Select sample
            ID_dir = (self.ct_dirs[index]).replace('\\','/')
            ID = ID_dir.split('/')[-1]

            # Get images
            img_file = ID_dir + '/' + 'ct_axial.nii'
            ct_vol = self.load_slices_vol(img_file)
            ct_vol_norm = self.normalise_images(ct_vol)

            if self.isTraining:
                ct_vol_aug = self.augment_data(ct_vol_norm, ct_vol_norm.shape[0])
            else:
                ct_vol_aug = ct_vol_norm[:]

            # Get labels
            labels_all_f = list(self.labels_df.loc[ID])
            labels_all_f = np.asarray(labels_all_f)
            labels = labels_all_f[self.feature_ids]

            # Convert to tensor
            img_torch = torch.from_numpy(ct_vol_aug).float()
            labels_torch = torch.from_numpy(labels).float()

            if self.isTraining:
                return img_torch, labels_torch
            else:
                return img_torch, labels_torch, ID