import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from dataio.dataset_lidc import DatasetLIDC
from model.model import Network
from utils.utils import *

import numpy as np

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
    if config.resume_epoch == None:
        make_new = True 
    else:
        make_new = False
    timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir, config.fold_id)
    experiment_path = 'experiments' + '/' + timestamp
    make_dir(experiment_path + '/' + json_opts.experiment_dirs.model_dir)

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
    train_dataset = DatasetLIDC(json_opts.data_source, config.fold_id, fold_mean, fold_std, 
                                json_opts.data_params.feature_ids, isTraining=True)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=json_opts.training_params.batch_size, 
                              shuffle=True, num_workers=num_workers)

    n_train_examples = len(train_loader)
    logging.info("Total number of training examples: %d" %n_train_examples)

    # Optimiser
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=json_opts.training_params.learning_rate, 
                                 betas=(json_opts.training_params.beta1, 
                                        json_opts.training_params.beta2),
                                 weight_decay=json_opts.training_params.l2_reg_alpha)

    # Resume training or train from scratch
    if config.resume_epoch != None:
        initial_epoch = config.resume_epoch
    else:
        initial_epoch = 0

    # Restore model
    if config.resume_epoch != None:
        load_path = experiment_path + '/' + json_opts.experiment_dirs.model_dir + "/epoch_%d.pth" %(config.resume_epoch)
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        assert(epoch == config.resume_epoch)
        print("Resume training, successfully loaded " + load_path)

    logging.info("Begin training")

    model = model.train()

    for epoch in range(initial_epoch, json_opts.training_params.total_epochs):
        epoch_train_loss = 0.
        epoch_mae_all = 0.

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # Permute channels axis to batch axis
            batch_x = batch_x.permute(1,0,2,3)

            # Transfer to GPU
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            y_pred, y_pred_aux, _, _ = model(batch_x)

            # Optimisation
            aux_loss = criterion(y_pred_aux, batch_y).squeeze()
            pred_loss = criterion(y_pred, batch_y).squeeze()
            loss = json_opts.training_params.aux_lambda*aux_loss + pred_loss
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.detach().cpu().numpy()
            epoch_mae_all += np.absolute(batch_y.squeeze().detach().cpu().numpy() - y_pred.squeeze().detach().cpu().numpy())

        # Save model
        if (epoch % json_opts.save_freqs.model_freq) == 0:
            save_path = experiment_path + '/' + json_opts.experiment_dirs.model_dir + "/epoch_%d.pth" %(epoch+1)
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_path)
            logging.info("Model saved: %s" % save_path)
                
        # Print training losses and performance every epoch
        print('Epoch[{}/{}], total loss:{:.4f}'.format(epoch+1, json_opts.training_params.total_epochs, epoch_train_loss))
        print('MAE', np.around(epoch_mae_all/n_train_examples, 5), np.mean(epoch_mae_all/n_train_examples))

    logging.info("Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', default='configs/config.json', type=str,
                        help='config file path')
    parser.add_argument('--resume_epoch', default=None, type=int,
                        help='resume training from this epoch, set to None for new training')
    parser.add_argument('--fold_id', default=1, type=int,
                        help='cross-validation fold')

    config = parser.parse_args()
    main(config)
