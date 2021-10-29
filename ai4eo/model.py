#!/usr/bin/env python

try: 
    import nni
except ImportError:
    pass

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import argparse
import os
import time
import datetime
from collections import defaultdict
import copy

import numpy as np
from sklearn.metrics import matthews_corrcoef

from eolearn.core import LoadTask

import eotasks
from srresnet_ai4eo import ConvolutionalBlock, SubPixelConvolutionalBlock, ResidualBlock, SRResNet 

SCALE = 4


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
        # band specification
        # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
        band_names = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'] # from starter notebook
        band_wavelength = [443, 490, 560, 665, 705, 740, 783, 842, 865, 940, 1610, 2190] # nm
        band_spatial_resolution = [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 20, 20] # m

        # read from args.processed_data_dir
        # division in train / valid or test
        # all available eopatches
        f_patches = os.listdir(args.processed_data_dir)
        assert args.fixed_random_seed # else the shuffle needs to go somewhere else

        np.random.shuffle(f_patches) 

        if flag=='train':
            f_patches = f_patches[args.n_valid_patches:]
        elif flag=='valid':
            f_patches = f_patches[:args.n_valid_patches]
            print('EOPatches used for validation:', f_patches)
        else:
            raise ValueError("not implemented: ", flag)

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

        min_patches = 100

        for patch in large_patches:
            min_patches = min(len(patch.data['BANDS']), min_patches)
            sp = eo_sample.execute(patch)
            small_patches.extend(sp)

        print(f'creating {len(small_patches)} small patches from {len(large_patches)} patches in {time.time()-start_time:.1f} seconds')
        print(f'minimum time frames: {min_patches}')

        # subsample time frame TODO
        tidx = list(range((args.n_time_frames+1)//2)) + list(range(-1*(args.n_time_frames//2), 0))
        #tidx = list(range(args.n_time_frames))
        print(f'selecting the first N//2 and the last N//2 time stamps: {tidx}')

        # subsample bands and other channels
        print('-- selecting bands --')
        for band in args.bands:
            print(f'  {band}')
        print('-- selecting normalized indices --')
        for index in args.indices:
            print(f'  {index}')

        lowres = []
        target = []
        weight = []
               
        for patch in small_patches:
            #print(f"time indices: {len(patch.data['BANDS'])}")
            x = []
            for ix in tidx: # outer most group: time index
                for band in args.bands:
                    band_ix = band_names.index(band)
                    if len(patch.data['BANDS']) > ix:
                        xx = patch.data['BANDS'][ix][:, :, band_ix]
                    else:
                        xx = patch.data['BANDS'][0][:, :, band_ix]
                    x.append(xx.astype(np.float32))
                for index in args.indices:
                    if len(patch.data['BANDS']) > ix:
                        xx = patch.data[index][ix]
                    else:
                        xx = patch.data[index][0]
                    x.append(xx.astype(np.float32).squeeze())

            y = patch.mask_timeless['CULTIVATED']
            ytf = np.sum(y) / len(y.flatten())
            #print(f'Target fraction: {100*ytf:.1f} %')
            if ytf < args.min_true_fraction:
                continue

            w = patch.data_timeless['WEIGHTS'] 
           
            # add rotated images
            for k in range(4):
                x_work = x.copy()
                if k>=1:
                    x = [np.rot90(x_work[i], k=k) for i in range(len(x))]
                    x = np.stack(x)
                lowres.append(np.array(x))
                if k>=1:
                    y = y.swapaxes(1,2)
                    y = y.swapaxes(0,2)
                    y = np.rot90(y, k=k)
                y = y.swapaxes(0,2)
                y = y.swapaxes(1,2)
                target.append(y.astype(np.float32))
                if k>=1:
                    w = w.swapaxes(1,2)
                    w = w.swapaxes(0,2)
                    w = np.rot90(w, k=k) 
                w = w.swapaxes(0,2)
                w = w.swapaxes(1,2)
                weight.append(w.astype(np.float32))
            

        # BANDS: time_idx * S * S * band_idx

        self.lowres = np.stack(lowres) # all input features
        self.target = np.concatenate(target) # all input features
        self.weight = np.concatenate(weight) # all input features
        print(f'{flag} dataset shapes: lowres = {self.lowres.shape}, target = {self.target.shape}')

    def __len__(self):
        return self.lowres.shape[0]

    def __getitem__(self, idx):
        return self.lowres[idx], self.target[idx], self.weight[idx]


# Main function
def main(args):

    def predict(inputs, target, weight, model, alpha=1, eval_=True):
        """Runs the prediction for a given model on data. Returns the loss together with the predicted
        values as numpy arrays."""

        device = model.get_device()
        inputs = inputs.to(device)
        target = target.to(device)
        weight = weight.to(device)

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

        # use weighted cross entropy loss with two classes
        S = target.shape[-1]
        target = target.reshape((-1, S*S))
        pred   = pred.reshape((-1, S*S))
        weight = weight.reshape((-1, S*S))
        loss1 = F.binary_cross_entropy(pred, target, weight=weight)

        # count pixels
        loss2 = (torch.sum(pred) - torch.sum(target)) / S / S

        print(loss1, loss2)
        loss = alpha * loss1 + (1 - alpha) * loss2

        return loss, pred_values

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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=8)
    # instantiate the model
    args.input_channels = args.n_time_frames*(len(args.bands)+len(args.indices))
    model = SRResNet(args)
    if torch.cuda.is_available():
        model = model.cuda()
    device = model.get_device()
    print(f'\nDevice {model.get_device()}')
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # training
    best_loss = np.inf
    best_mcc = -np.inf
    best_epoch = 0
    patience_count = 0
    # history
    all_train_losses = []
    all_valid_losses = []

    for epoch in range(args.max_epochs):
        # train
        model.train()
        start_time = time.time()
        print(f'\nEpoch: {epoch}')
        train_losses = []
        for idx, (inputs, target, weight) in enumerate(train_loader):
            loss, _ = predict(inputs, target, weight, model, args.alpha, eval_=False)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.debug:
                break
        train_loss = np.mean(np.array(train_losses))
        all_train_losses.append(train_loss)
        print(f'Training took {(time.time() - start_time) / 60:.2f} minutes, train_loss: {train_loss:.4f}')
        start_time = time.time()
        # validation
        model = model.eval()
        valid_losses, preds ,targets, weights = [], [], [], []
        for idx, (inputs, target, weight) in enumerate(valid_loader):
            loss, pred = predict(inputs, target, weight, model, args.alpha, eval_=True)
            valid_losses.append(loss.detach().cpu().numpy())
            preds.append(pred)
            targets.append(target)
            weights.append(weight)
        valid_loss = np.mean(np.array(valid_losses))
        all_valid_losses.append(valid_loss)
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        weights = np.concatenate(weights, axis=0)
        if args.nni:
            mcc = calc_evaluation_metric(targets, preds.round().astype(np.float32), weights)
        print(f'Validation took {(time.time() - start_time) / 60:.2f} minutes, valid_loss: {valid_loss:.4f}')
        # nni
        if args.nni:
            nni.report_intermediate_result(mcc)
        # early stopping
        if valid_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_preds = preds
            cast_best_preds = (best_preds > 0.5).astype(np.float32)
            best_valid_loss = valid_loss
            if args.nni:
                best_mcc = mcc
            else:
                best_mcc = calc_evaluation_metric(targets, cast_best_preds, weights)
            patience_count = 0
        else:
            patience_count += 1

        if patience_count == args.patience:
            print(f'no improvement for {args.patience} epochs, early stopping')
            break

        if epoch % args.checkpoint_epoch == 0:
            save_model_path = os.path.join(args.target_dir, f'epoch_{epoch}_model.pt')
            torch.save(model.state_dict(), save_model_path)
    if args.nni:
        nni.report_final_result(best_mcc)
    # save best model and TODO predictions to disk
    save_model_path = os.path.join(args.target_dir, 'best_model.pt')
    torch.save(best_model.state_dict(), save_model_path)
    print(f'saved best model to {save_model_path}')
    # save training history
    with open(os.path.join(args.target_dir, 'train_losses.txt'), 'w') as f:
        for tl in all_train_losses:
            f.write(f'{tl:.4f}\n')
    with open(os.path.join(args.target_dir, 'valid_losses.txt'), 'w') as f:
        for vl in all_valid_losses:
            f.write(f'{vl:.4f}\n')

def calc_evaluation_metric(target, pred, weight):
    '''
    calculate evaluation metric MCC as given in the task
    '''
    b_size = target.shape[0]
    mcc = []
    for i in range(b_size):
        mcc.append(matthews_corrcoef(target[i,...].flatten(), pred[i,...].flatten(), sample_weight=weight[i,...].flatten()))
    MCC = np.stack(mcc)
    MCC = np.mean(MCC, axis=0)
    print(f'evaluation metric MCC: {MCC:.4f}')
    return MCC


def add_nni_params(args):
    args_nni = nni.get_next_parameter()
    assert all([key in args for key in args_nni.keys()]), 'need only valid parameters'
    args_dict = vars(args)
    # cast params that should be int to int if needed (nni may offer them as float)
    args_nni_casted = {key:(int(value) if type(args_dict[key]) is int else value) for key, value in args_nni.items()}
    args_dict.update(args_nni_casted)
    # adjust paths of model and prediction outputs so they get saved together with the other outputs
    nni_output_dir = os.path.expandvars('$NNI_OUTPUT_DIR')
    for param in ['target_dir']:
        nni_path = os.path.join(nni_output_dir, os.path.basename(args_dict[param]))
        args_dict[param] = nni_path
    return args

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-data-dir', type=str, default='/work/shared_data/2021-ai4eo/dev_data/default/')
    parser.add_argument('--target-dir', type=str, default='.')
    parser.add_argument('--fixed-random-seed', action='store_true', default=True, help='fixed random seed numpy / torch') 
    parser.add_argument('--inference', action='store_true', default=False, help='run test set inference')
    parser.add_argument('--nni', action='store_true', default=False)
    # data set hyperparameters
    parser.add_argument('--debug', action='store_true', default=False, help='skip after first batch in training')
    parser.add_argument('--n-valid-patches', type=int, default=10, help='Number of EOPatches selected for validation')
    parser.add_argument('--s2-length', type=int, default=32, help='Cropped EOPatch samples side length')
    parser.add_argument('--n-time-frames', type=int, default=1, help='Number of time frames in EOPatches')
    parser.add_argument('--s2-random', action='store_true', 
                        help='Randomly select overlapping patches (else: systematically select non overlapping patches')
    parser.add_argument('--n-s2', type=int, default=10, help='number of EOPatches to subsample')
    parser.add_argument('--bands', type=str, nargs='*', default=[], help='Sentinel band names (--> starter notebook')
    parser.add_argument('--indices', type=str, nargs='*', default=['NDVI'], choices=["NDVI", "NDWI", "NDBI"])
    # network and training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--filters', type=int, default=8)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=6, help='early stopping patience, -1 for no early stopping')
    parser.add_argument('--checkpoint-epoch', type=int, default=1000, help='checkpoint every nth epoch')
    # the scaling factor (for the Generator), the input LR images will be downsampled from the target HR images by this factor 
    parser.add_argument('--scaling_factor', type=int, default=4)
    # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
    parser.add_argument('--n-channels', type=int, default=64)
    # nr of input channels: 1: no bands, >1 bands B01, B02, etc included
    #parser.add_argument('--input_channels', type=int, default=3)  # number of input channels, default for RGB image: 3
    # kernel size of the first and last convolutions which transform the inputs and outputs
    parser.add_argument('--large_kernel_size', type=int, default=9)
    # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
    parser.add_argument('--small_kernel_size', type=int, default=3)
    # minimum fraction of true samples (avoid empty samples?)
    parser.add_argument('--min-true-fraction', type=float, default=0, help='minimum fraction of positive pixels in target') 
    parser.add_argument('--n-blocks', type=int, default=16) # number of residual blocks
    parser.add_argument('--alpha', type=float, default=1, help='lagrange multiplier for the second loss pushing for pixel count conservation')
    args = parser.parse_args()

    if args.nni:
        args = add_nni_params(args)

    print('\n*** begin args key / value ***')
    for key, value in vars(args).items():
        print(f'{key:20s}: {value}')
    print('*** end args key / value ***\n')

    main(args)
