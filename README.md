# An attention-enhanced cross-task network to analyse lung nodule attributes in CT images

Fu, X., Bi, L., Kumar, A., Fulham, M., & Kim, J. (2022). An attention-enhanced cross-task network to analyse lung nodule attributes in CT images. Pattern Recognition, 126, 108576.
https://doi.org/10.1016/j.patcog.2022.108576 

Implementation of the proposed deep learning model in Python and PyTorch. 

![Alt text](FigS1.png?raw=true)

## Requirements

Python ``3.7.8`` to ``3.8.8``

PyTorch ``1.5.0`` to ``1.10.2``

Run ``pip install -r requirements.txt`` in your shell to install additional dependencies.

## Data

#### CT Images:
- Nodules resized to 64x64 (see paper for details).
- .nii files containing all the slices for each nodule.
- For example: ``LIDC-IDRI-0001-1/ct_axial.nii``

#### Labels:
- .csv file containing the ground truth attribute ratings for all nodules.
- File contains 10 columns. First column is nodule IDs. Subsequent columns are for the 9 attributes (subtlety, internal structure, calcification, sphericity, margin, lobulation, spiculation, texture, and malignancy).
- Nodule IDs are in the format ``LIDC-IDRI-0001-1``, ``LIDC-IDRI-0002-1``, etc.
- Ratings are normalised to [0, 1]

#### Cross-validation fold splits:
- .txt files each containing a list of nodule IDs for different cross-validation folds.
- For example for fold 1: `1_train.txt`, and `1_test.txt`.

## Operation

Modify hyperparameters and locations of data files in the .json config file inside ``/configs``.

Train the model using ``train.py``.

Additional arguments for training:
- ``--config_file`` — path to config file
- ``--fold_id`` — cross-validation fold number
- ``--resume_epoch`` — ``None`` for train from scratch, or int number (e.g., 5) to resume training from saved model

Test the model using ``test.py``.

Additional arguments for testing:
- ``--config_file`` — path to config file
- ``--fold_id`` — cross-validation fold number
- ``--test_epoch`` — which epoch to test (int number, -1 for latest saved model, or -2 for all saved models)
