#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import argparse
import os
import time
from collections import defaultdict
import copy

import numpy as np
from sklearn.metrics import matthews_corrcoef

from eolearn.core import LoadTask

import eotasks

SCALE = 4

#from pytorch_lightning.metrics import Metric
#from pytorch_lightning.metrics.functional.classification import (
#    stat_scores_multiple_classes
#)
#
## TODO
## https://github.com/scikit-learn/scikit-learn/blob/f3067cb6201101df6d4f12f25fe488f6b35006cc/sklearn/metrics/_classification.py
#
#
#
#class MCC(Metric):
#    r"""
#    Adapted from https://gist.github.com/abhik-99/7564fdac4ede90fc7b99ef91abd64041
#
#    Computes `Mathews Correlation Coefficient <https://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_:
#    Forward accepts
#    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
#    - ``target`` (long tensor): ``(N, ...)``
#    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
#    This is the case for binary and multi-label logits.
#    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.
#    Args:
#        labels: Classes in the dataset.
#        pos_label: Treats it as a binary classification problem with given label as positive.
#    """
#    def __init__(
#        self,
#        labels,
#        pos_label = None, 
#        compute_on_step = True,
#        dist_sync_on_step = False,
#        process_group = None,
#    ):
#        super().__init__(
#            compute_on_step=compute_on_step,
#            dist_sync_on_step=dist_sync_on_step,
#            process_group=process_group,
#        )
#
#        self.labels = labels
#        self.num_classes = len(labels)
#        self.idx = None
#
#        if pos_label is not None:
#          self.idx = labels.index(pos_label)
#
#        self.add_state("matthews_corr_coef", default=torch.tensor(0), dist_reduce_fx="mean")
#        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
#
#    def update(self, preds: torch.Tensor, target: torch.Tensor):
#        """
#        Update state with predictions and targets.
#        Args:
#            preds: Predictions from model
#            target: Ground truth values
#        """
#        tps, fps, tns, fns, _ = stat_scores_multiple_classes(
#            pred=preds, target=target, num_classes=self.num_classes)
#        
#        if self.idx is not None:
#          tps, fps, tns, fns = tps[self.idx], fps[self.idx], tns[self.idx], fns[self.idx]
#        
#        numerator = (tps * tns) - (fps * fns)
#        denominator = torch.sqrt(((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns)))
#        
#        self.matthews_corr_coef = numerator / denominator
#        #Replacing any NaN values with 0
#        self.matthews_corr_coef[torch.isnan(self.matthews_corr_coef)] = 0 
#        
#        self.total += 1
#
#    def compute(self):
#        """
#        Computes Matthews Correlation Coefficient over state.
#        """
#        return self.matthews_corr_coef / self.total

# Data set
class EODataset(Dataset):
    def __init__(self, flag, args):
        '''
        flag : train / valid / test
        args : argparse namespace
        '''

        if flag=='test':
            print(f'not implemented: {flag}')
            return

        # read from args.processed_data_dir
        # division in train / valid or test
        # all available eopatches
        f_patches = os.listdir(args.processed_data_dir)
        assert args.fixed_random_seed # else the shuffle needs to go somewhere else

        np.random.shuffle(f_patches) 

        if flag=='train':
            f_patches = f_patches[args.n_valid_patches:]
        else:
            f_patches = f_patches[:args.n_valid_patches]

        # load patches
        eo_load = LoadTask(path=args.processed_data_dir)
        large_patches = []

        start_time = time.time()
        
        for f_patch in f_patches:
            eopatch = eo_load.execute(eopatch_folder=f_patch)
            large_patches.append(eopatch)

        print(f'loading {flag} data took {time.time()-start_time:.1f} seconds')

        # subsample to smaller images
        eo_sample = eotasks.SamplePatchletsTask(s2_patchlet_size=args.s2_length, 
                                                num_samples=args.n_s2, 
                                                random_mode=args.s2_random)

        small_patches = []

        start_time = time.time()

        for patch in large_patches:
            sp = eo_sample.execute(patch)
            small_patches.extend(sp)

        print(f'creating {len(small_patches)} small patches from {len(large_patches)} patches in {time.time()-start_time:.1f} seconds')

        # subsample time frame TODO
        tidx = 0
        print(f'selecting the very first time stamp')

        # subsample bands and other channels TODO
        print(f'selecting NDVI channel only')

        lowres = []
        target = []
        weight = []

        for patch in small_patches:
            x = patch.data['NDVI'][tidx]
            lowres.append(x.astype(np.float32))
            y = patch.mask_timeless['CULTIVATED']
            target.append(y.astype(np.float32))
            w = patch.data_timeless['WEIGHTS']
            weight.append(w.astype(np.float32))

        # BANDS: time_idx * S * S * band_idx

        self.lowres = np.array(lowres) # all input features
        self.target = np.array(target) # the target map
        self.weight = np.array(weight) # the pixel weights

        print(f'{flag} dataset shapes: lowres = {self.lowres.shape}, target = {self.target.shape}')

    def __len__(self):
        return self.lowres.shape[0]

    def __getitem__(self, idx):
        return self.lowres[idx], self.target[idx], self.weight[idx]

# Model definition
class EOModel(nn.Module):
    def __init__(self, args):
        # stub: add proper architecture
        super().__init__()
        self.lr1 = nn.Linear(args.s2_length * args.s2_length, 16)
        self.lr2 = nn.Linear(16, SCALE * SCALE * args.s2_length * args.s2_length)

    def forward(self, x):
        # stub: returns batch_size random images with correct dimension
        x = torch.flatten(x, 1)
        x = F.relu(self.lr1(x))
        x = self.lr2(x)
        x = torch.reshape(x, (-1, SCALE * args.s2_length, SCALE * args.s2_length))
        return x

    # Other useful functions
    def get_device(self):
        '''Return gpu if available, else cpu'''
        if torch.cuda.is_available():
            return 'cuda:0'
        else:
            return 'cpu'


# Main function
def main(args):

    def predict(inputs, target, weight, model, eval_=True):
        """Runs the prediction for a given model on data. Returns the loss together with the predicted
        values as numpy arrays."""

        device = model.get_device()

        inputs = inputs.to(device)
        target = target.to(device)

        if eval_:
            with torch.no_grad():
                pred = model(inputs)
        else:
            pred = model(inputs)

        # get the predicted values off the GPU / off torch
        if torch.cuda.is_available():
            pred_values = pred.cpu().detach().numpy()
        else:
            pred_values = pred.detach().numpy()

        # TODO pred / target dimension
        pred = torch.reshape(pred, (len(inputs), -1))
        target = torch.reshape(target, (len(inputs), -1))
        weight = torch.reshape(weight, (len(inputs), -1))
        print(pred.shape)
        print(pred)
        print(target)
        losses = matthews_corrcoef(target.detach().numpy(), pred.detach().numpy(), sample_weight=weight.detach().numpy())

        return losses, pred_values

    # start the program
    if args.fixed_random_seed:
        np.random.seed(2021)
        torch.manual_seed(1407)

    if args.inference:
        print('not implemented: inference')
        return

    # construct the dataset
    train_dataset = EODataset('train', args)
    valid_dataset = EODataset('valid', args)
    # construct the dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    # instantiate the model
    model = EOModel(args)
    device = model.get_device()
    print(f'\nDevice {device}\n')
    # optimizer
    #loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # training
    best_loss = np.inf
    best_epoch = 0
    patience_count = 0

    for epoch in range(args.max_epochs):
        # train
        model.train()
        start_time = time.time()
        print(f'\nEpoch: {epoch}')
        train_losses = []
        for idx, (inputs, target, weight) in enumerate(train_loader):
            loss, _ = predict(inputs, target, weight, model, eval_=False)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = np.mean(np.array(train_losses))
        # validation
        model = model.eval()
        valid_losses, preds = [], []
        for idx, (inputs, target, weight) in enumerate(valid_loader):
            loss, pred = predict(inputs, target, weight, model, eval_=True)
            valid_losses.append(loss)
            preds.append(pred)
        valid_loss = np.mean(np.array(valid_losses))
        pred = np.concatenate(pred, axis=0)
        print(f'Epoch took {(time.time() - start_time) / 60:.2f} mins')
        print(f'train loss: {train_loss}, valid_loss: {valid_loss}')
        # nni
        if args.nni:
            nni.report_intermediate_result(valid_loss)
        # early stopping
        if valid_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_preds = preds
            best_loss = valid_loss
            patience_count = 0
        else:
            patience_count += 1

        if patience_count == args.patience:
            print(f'no improvement for {args.patience} epochs, early stopping')
            break

    if args.nni:
        nni.report_final_result(best_valid_loss)
    # TODO save best model and predictions to disk
    save_model_path = os.path.join(args.target_dir, 'best_model.pt')
    torch.save(best_model.state_dict(), save_model_path)
    print(f'saved best model to {save_model_path}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-data-dir', type=str, default='/work/shared_data/2021-ai4eo/dev_data/default/')
    parser.add_argument('--target-dir', type=str, default='.')
    parser.add_argument('--n-processes', type=int, default=4, help='Processes for EOExecutor')
    parser.add_argument('--n-valid-patches', type=int, default=10, help='Number of EOPatches selected for validation')
    parser.add_argument('--s2-length', type=int, default=32, help='Cropped EOPatch samples side length')
    parser.add_argument('--n-time-frames', type=int, default=1, help='Number of time frames in EOPatches')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the output files')
    parser.add_argument('--s2-random', action='store_true', 
                        help='Randomly select overlapping patches (else: systematically select non overlapping patches')
    parser.add_argument('--n-s2', type=int, default=10, help='number of EOPatches to subsample')
    parser.add_argument('--fixed-random-seed', action='store_true', default=True, help='fixed random seed numpy / torch') 
    parser.add_argument('--inference', action='store_true', default=False, help='run test set inference')
    parser.add_argument('--nni', action='store_true', default=False)
    # network and training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=6, help='early stopping patience, -1 for no early stopping')

    args = parser.parse_args()

    print('\n*** begin args key / value ***')
    for key, value in vars(args).items():
        print(f'{key:20s}: {value}')
    print('*** end args key / value ***\n')

    main(args)
