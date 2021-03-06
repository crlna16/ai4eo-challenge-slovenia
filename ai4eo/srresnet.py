import os
try:
    import nni
except ImportError:
    pass
import numpy
import math
import h5py
import argparse
import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

class SRDataset(Dataset):
    def __init__(self, data_path, transform=None):
        '''
        data: image
        transform: optional transforms applied to the image
        '''
        start_time = time.time()
        print(f'read images {data_path} ...')
        # read data
        h5_file = h5py.File(data_path, 'r')
        self.X = h5_file['X'][:]
        self.y = h5_file['y'][:]
        self.transform = transform
        print(f'loading images took {start_time - time.time():.4f}sec')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

class SRDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            sr_train = os.path.join(args.data_dir, 'train.h5')
            sr_valid = os.path.join(args.data_dir, 'valid.h5')

            self.train_dataset = SRDataset(sr_train, transform=self.transform)
            self.valid_dataset = SRDataset(sr_valid, transform=self.transform)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print(f'train_datloader: {next(iter(train_dataloader))[0].shape}')
        print(f'train_dataloader: {next(iter(train_dataloader))[1].shape}')
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.valid_dataset, batch_size=args.batch_size)
        print(f'val_dataloader: {next(iter(val_dataloader))[0].shape}')
        print(f'val_dataloader: {next(iter(val_dataloader))[1].shape}')
        return val_dataloader

class ConvolutionalBlock(pl.LightningModule):
    '''
    Convolutional block: Convolution, BatchNorm, Activation
    credits: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/blob/master/models.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: stride
        :param batch_norm: include a BN layer?
        :param activation: Type of activation; None if none
        '''
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # container, that will hold the layers in this convolutional block
        layers = list()
        # convolutional layer
        layers.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                    stride=stride, padding=kernel_size // 2)
                )
        # batch normalization, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # activation layer, if wanted
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # put together the convolutional block as a sequence of the layers
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        '''
        Forward propagation

        :param input: input images, a tensor of size (N, in_channels, w, h)
        :return: output images, a tensor of size (N, out_channels, w, h)
        '''
        output = self.conv_block(input) #(N, out_channels, w, h)
        return output

class SubPixelConvolutionalBlock(pl.LightningModule):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """
    def __init__(self, args, scaling_factor=2):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()
        # convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        self.conv = nn.Conv2d(in_channels=args.n_channels, out_channels=args.n_channels * (scaling_factor ** 2),
                            kernel_size=args.small_kernel_size, padding=args.small_kernel_size // 2)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input_):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input_)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class ResidualBlock(pl.LightningModule):
    """
    A residual block, comprising two convolutional blocks with a residual connection across them.
    """
    def __init__(self, args):
        """
        :param kernel_size: kernel size
        :param n_channels: number of input and output channels (same because the input must be added to the output)
        """
        super(ResidualBlock, self).__init__()

        # first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=args.n_channels, out_channels=args.n_channels, 
                kernel_size=args.small_kernel_size, batch_norm=True, activation='PReLu')

        # second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=args.n_channels, out_channels=args.n_channels,
                kernel_size=args.small_kernel_size, batch_norm=True, activation=None)

    def forward(self, input_):
        """
        Forward propagation
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: output images, a tensor of size (N, n_channels, w, h)
        """
        residual = input_ # (N, n_channels, w, h)
        output = self.conv_block1(input_) # (N, n_channels, w, h)
        output = self.conv_block2(output) # (N, n_channels, w, h)
        output = output + residual # (N, n_channels, w, h)

        return output
    

class SRResNet(pl.LightningModule):
    '''
    The SRResNet, as defined in the paper.
    credits: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/blob/master/models.py
    '''
    def __init__(self, args):
        """
        :param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        :param small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
        :param n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
        :param n_blocks: number of residual blocks
        :param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
        """
        self.args = args
        super(SRResNet, self).__init__()

        # Scaling factor must be 2, 4 or 8
        scaling_factor = int(args.scaling_factor)
        assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

        # First convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=args.input_channels, out_channels=args.n_channels, 
                kernel_size=args.large_kernel_size,
                batch_norm=False, activation='PReLu')

        # Sequence of residual blocks
        self.residual_blocks = nn.Sequential(
                *[ResidualBlock(args) for i in range(args.n_blocks)]
                )

        # Another convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=args.n_channels, out_channels=args.n_channels,
                kernel_size=args.small_kernel_size, batch_norm=True, activation=None)

        # Upscaling: by sub-pixel convolution, each such block upscaling by a factor of 2
        n_subpixel_convolutional_blocks = int(math.log2(args.scaling_factor))
        print(f'times subpixel: {n_subpixel_convolutional_blocks}')
        self.subpixel_convolutional_blocks = nn.Sequential(
                *[SubPixelConvolutionalBlock(args, scaling_factor=2) for i in range(n_subpixel_convolutional_blocks)]
                )

        # Last convolutional block
        self.conv_block3 = ConvolutionalBlock(in_channels=args.n_channels, out_channels=1,
                kernel_size=args.large_kernel_size, batch_norm=False, activation='Tanh')

        # Final sigmoid layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, lr_imgs):
        """
        Forward propagation

        :param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
        :return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs) # (N, input_channels, w, h)
        residual = output # (N, n_channels, w, h)
        output = self.residual_blocks(output) # (N, n_channels, w, h)
        output =self.conv_block2(output) # (N, n_channels, w, h)
        output = output + residual
        output = self.subpixel_convolutional_blocks(output) # (N, n_channels, w * scaling factor, h * scaling factor)
        sr_imgs = self.conv_block3(output) # (N, 1, w * scaling factor, h * scaling factor)
        sr_imgs = self.sigmoid(sr_imgs)
        #sr_imgs = sr_imgs.round() # reduce to image containing only 1 and 0
        return sr_imgs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        return optimizer

    def mse_loss(self, y_hat, y):
        criterion = nn.MSELoss()
        loss = criterion(y_hat, y)
        return loss

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = self.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = self.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def predict_step(self, val_batch, batch_idx, dataloader_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.mse_loss(y_pred, y)
        return y_pred
    
class SRCallbacks(Callback):
    def __init__(self, args):
        self.args = args

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        print('\nValiddation epoch end:')
        for key, item in metrics.items():
            print(f'{key}: {item:.4}')

        if self.args.nni:
            nni.report_intermediate_result(float(metrics['val_loss']))

    def on_train_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        print(f'\Final validation loss:')
        for key, item in metrics.items():
            print(f'{key}: {item:.4}')
        if self.args.nni:
            nni.report_final_result(float(metrics['val_loss']))


def add_nni_params(args):
    args_nni = nni.get_next_parameter()
    assert all([key in args for key in args_nni.keys()]), 'need only valid parameters'
    args_dict = vars(args)
    # cast params that should be int to int if needed (nni may offer them as float)
    args_nni_casted = {key:(int(value) if type(args_dict[key]) is int else value)
                       for key, value in args_nni.items()}
    args_dict.update(args_nni_casted)

    # adjust paths of model and prediction outputs so they get saved together with the other outputs
    nni_output_dir = os.path.expandvars('$NNI_OUTPUT_DIR')
    for param in ['save_model_path', 'output_path']:
        nni_path = os.path.join(nni_output_dir, os.path.basename(args_dict[param]))
        args_dict[param] = nni_path
    return args



def make_predictions(model, dataloader, trainer, args):
    y_pred = trainer.predict(model, dataloader)
    y_pred = torch.cat(y_pred).detach().cpu().numpy()

    h5_file = h5py.File(args.output_path, 'w')
    chunk_size =  y_pred.shape[0]
    dset = h5_file.create_dataset('y_pred',
                                  shape=y_pred.shape,
                                  chunks=(chunk_size,) + y_pred.shape[1:],
                                  fletcher32=True,
                                  dtype='float32')
    dset[:] = y_pred
    h5_file.attrs['timestamp'] = str(datetime.datetime.now())
    h5_file.close()
    print(f'saved high resolution images to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-path', type=str, default='data/best_hr_predictions.h5')
    parser.add_argument('--save-model-path', type=str, default='saved_models')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--scaling_factor', type=int, default=4) # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
    parser.add_argument('--n_channels', type=int, default=64)  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks# number of residual blocks
    parser.add_argument('--input_channels', type=int, default=3)  # number of input channels, default for RGB image: 3
    parser.add_argument('--large_kernel_size', type=int, default=9) # kernel size of the first and last convolutions which transform the inputs and outputs
    parser.add_argument('--small_kernel_size', type=int, default=3) # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks 
    parser.add_argument('--n_blocks', type=int, default=16) # number of residual blocks
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--nni', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.nni:
        args = add_nni_params(args)

    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print()

    # callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=args.save_model_path, filename='best_model')#'{epoch}-{val_loss:.2f}-{other_metric:.2f}')
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")

    model = SRResNet(args)
    data_module = SRDataModule(args)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, SRCallbacks(args)], max_epochs=args.n_epochs)
    trainer.fit(model, data_module)

    # make predictions on validation set
    valid_model = SRResNet.load_from_checkpoint(os.path.join(args.save_model_path,'best_model.ckpt'), args=args).eval().cuda()
    make_predictions(valid_model, data_module.val_dataloader(), trainer, args)

     
