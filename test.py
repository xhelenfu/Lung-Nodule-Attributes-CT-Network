import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

from dataio.dataset_lidc import DatasetLIDC
from model.model import Network
from utils.utils import *

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main(config):

    json_opts = json_file_to_pyobj(config.config_file)

    if json_opts.training_params.batch_size > 1:
        sys.exit('Batch size > 1 not supported')

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create experiment directories
    make_new = False
    timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir, config.fold_id)
    experiment_path = 'experiments' + '/' + timestamp
    model_dir = experiment_path + '/' + json_opts.experiment_dirs.model_dir
    test_output_dir = experiment_path + '/' + json_opts.experiment_dirs.test_output_dir
    make_dir(test_output_dir)

    fold_mean = json_opts.data_params.fold_means[config.fold_id-1]
    fold_std = json_opts.data_params.fold_stds[config.fold_id-1]

    # Set up the model
    logging.info("Initialising model")
    model_opts = json_opts.model_params
    n_features = len(json_opts.data_params.feature_ids)
    model = Network(model_opts, n_features)
    model = model.to(device)

    # Dataloader
    logging.info("Preparing data")
    num_workers = json_opts.data_params.num_workers
    test_dataset = DatasetLIDC(json_opts.data_source, config.fold_id, fold_mean, fold_std, 
                               json_opts.data_params.feature_ids, isTraining=False)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=1, 
                             shuffle=False, num_workers=num_workers)

    n_test_examples = len(test_loader)
    logging.info("Total number of testing examples: %d" %n_test_examples)

    # Get list of model files
    if config.test_epoch < 0:
        saved_model_paths, _ = get_files_list(model_dir, ['.pth'])
        saved_model_paths = sorted_alphanumeric(saved_model_paths)
        saved_model_epochs = [(os.path.basename(x)).split('.')[0] for x in saved_model_paths]
        saved_model_epochs = [x.split('_')[-1] for x in saved_model_epochs]
        if config.test_epoch == -2:
            saved_model_epochs = np.array(saved_model_epochs, dtype='int')
        elif config.test_epoch == -1:
            saved_model_epochs = np.array(saved_model_epochs[-1], dtype='int')
            saved_model_epochs = [saved_model_epochs]
    else:
        saved_model_epochs = [config.test_epoch]

    logging.info("Begin testing")

    mae_epochs = np.zeros((len(saved_model_epochs), n_features))
    mse_epochs = np.zeros((len(saved_model_epochs), n_features))
    mae_epochs_avg = np.zeros(len(saved_model_epochs))
    mse_epochs_avg = np.zeros(len(saved_model_epochs))
    
    for epoch_idx, test_epoch in enumerate(saved_model_epochs):

        gt_all = np.zeros((n_test_examples, n_features))
        pred_all = np.zeros((n_test_examples, n_features))

        # Restore model
        load_path = model_dir + "/epoch_%d.pth" %(test_epoch)
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        assert(epoch == test_epoch)
        print("Testing " + load_path)

        model = model.eval()

        # Write predictions to text file
        txt_path = test_output_dir + '/' + 'epoch_' + str(test_epoch) + '.txt'
        
        with open(txt_path, 'w') as output_file:

            for batch_idx, (batch_x, batch_y, ID) in enumerate(test_loader):

                # Permute channels axis to batch axis
                batch_x = batch_x.permute(1,0,2,3)

                # Transfer to GPU
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Forward pass
                y_pred, _, _, _ = model(batch_x)

                # Labels, predictions per example
                gt_all[batch_idx,:] = batch_y.squeeze().detach().cpu().numpy()[json_opts.data_params.feature_ids]
                pred_all[batch_idx,:] = y_pred.squeeze().detach().cpu().numpy()

                for f in range(n_features):
                    output_file.write(ID[0] + ' ' + str(f) + ' gt: ' + str(gt_all[batch_idx,f]) + ' predict: ' + str(pred_all[batch_idx,f]) + '\n')

            # Compute performance for all test examples this epoch
            mae_all = np.zeros(n_features)
            mse_all = np.zeros(n_features)

            # Convert back to original scale
            for f in range(n_features):
                if f == 1:
                    gt_all[:,f] = gt_all[:,f]*3+1
                    pred_all[:,f] = pred_all[:,f]*3+1
                elif f == 2:
                    gt_all[:,f] = gt_all[:,f]*5+1
                    pred_all[:,f] = pred_all[:,f]*5+1
                else:
                    gt_all[:,f] = gt_all[:,f]*4+1
                    pred_all[:,f] = pred_all[:,f]*4+1

                mae_all[f] = mean_absolute_error(gt_all[:,f], pred_all[:,f])
                mse_all[f] = mean_squared_error(gt_all[:,f], pred_all[:,f])

            output_file.write('Overall MAE and MSE \n')
            output_file.write(" ".join(map(str, np.around(mae_all, 5))))
            output_file.write('\n')
            output_file.write(" ".join(map(str, np.around(mse_all, 5))))

        # Store performances for each feature
        print('MAE:', np.around(mae_all, 5))
        mae_epochs[epoch_idx,:] = mae_all
        print('MSE:', np.around(mse_all, 5))
        mse_epochs[epoch_idx,:] = mse_all

        # Means for this epoch
        mae_epochs_avg[epoch_idx] = np.mean(mae_all)
        mse_epochs_avg[epoch_idx] = np.mean(mse_all)
        print('MAE mean: ', mae_epochs_avg[epoch_idx])
        print('MSE mean: ', mse_epochs_avg[epoch_idx])

    best_epoch = np.argmin(mae_epochs_avg)
    print('Best mae mean: epoch %d, mae %.7f, mse %.7f' %(saved_model_epochs[best_epoch], np.min(mae_epochs_avg), mse_epochs_avg[best_epoch]))
    print('MAEs:', np.around(mae_epochs[best_epoch,:], 5))
    print('MSEs:', np.around(mse_epochs[best_epoch,:], 5))

    logging.info("Testing finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', default='configs/config.json', type=str,
                        help='config file path')
    parser.add_argument('--test_epoch', default=-2, type=int,
                        help='test model from this epoch, -1 for last, -2 for all')
    parser.add_argument('--fold_id', default=1, type=int,
                        help='cross-validation fold')

    config = parser.parse_args()
    main(config)
