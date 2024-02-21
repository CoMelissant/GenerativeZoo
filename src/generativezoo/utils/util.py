from pathlib import Path
import numpy as np
import argparse


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def set_seed(seed: int = 42) -> None:
    """Set random seed for numpy.

    https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
    """
    rng = np.random.default_rng(seed)
    return rng

def parse_args_VanillaVAE():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample model')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argparser.add_argument('--latent_dim', type=int, default=128, help='latent dimension')
    argparser.add_argument('--hidden_dims', type=int, nargs='+', default=None, help='hidden dimensions')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--loss_type', type=str, default='mse', help='loss type', choices=['mse', 'ssim'])
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    return argparser.parse_args()

def parse_args_ConditionalVAE():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample model')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argparser.add_argument('--latent_dim', type=int, default=128, help='latent dimension')
    argparser.add_argument('--hidden_dims', type=int, nargs='+', default=None, help='hidden dimensions')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample and save frequency')
    return argparser.parse_args()

def parse_args_AdversarialVAE():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--test', action='store_true', default=False, help='test model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample model')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argparser.add_argument('--latent_dim', type=int, default=128, help='latent dimension')
    argparser.add_argument('--hidden_dims', type=int, nargs='+', default=None, help='hidden dimensions')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--gen_weight', type=float, default=0.002, help='generator weight')
    argparser.add_argument('--recon_weight', type=float, default=0.002, help='reconstruction weight')
    argparser.add_argument('--sample_and_save_frequency', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--discriminator_checkpoint', type=str, default=None, help='discriminator checkpoint path')
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    return argparser.parse_args()

def parse_args_VanillaSGM():
    argparser = argparse.ArgumentParser()
    # show choices: mnist | cifar10 | fashionmnist | chestmnist | octmnist | tissuemnist | pneumoniamnist | svhn
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample model')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--model_channels', type=int, nargs='+', default=[32, 64, 128, 256], help='model channels')
    argparser.add_argument('--atol', type=float, default=1e-6, help='absolute tolerance')
    argparser.add_argument('--rtol', type=float, default=1e-6, help='relative tolerance')
    argparser.add_argument('--eps', type=float, default=1e-3, help='smallest timestep for numeric stability')
    argparser.add_argument('--snr', type=float, default=0.16, help='signal to noise ratio')
    argparser.add_argument('--sample_timesteps', type=int, default=1000, help='number of sampling timesteps')
    argparser.add_argument('--embed_dim', type=int, default=256, help='embedding dimension')
    argparser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    argparser.add_argument('--sigma', type=float, default=25.0, help='sigma')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--num_steps', type=int, default=500, help='number of steps')
    argparser.add_argument('--sampler_type', type=str, default='EM', help='sampler type', choices=['EM', 'PC', 'ODE'])
    argparser.add_argument('--sample_and_save_freq', type=int, default=10, help='sample and save frequency')
    return argparser.parse_args()

def parse_args_DDPM():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample model')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--timesteps', type=int, default=300, help='number of timesteps')
    argparser.add_argument('--n_features', type=int, default = 64, help='number of features')
    argparser.add_argument('--init_channels', type=int, default = 32, help='initial channels')
    argparser.add_argument('--channel_scale_factors', type=int, nargs='+', default = [1, 2, 2], help='channel scale factors')
    argparser.add_argument('--resnet_block_groups', type=int, default = 8, help='resnet block groups')
    argparser.add_argument('--use_convnext', type=bool, default = True, help='use convnext (default: True)')
    argparser.add_argument('--convnext_scale_factor', type=int, default = 2, help='convnext scale factor (default: 2)')
    argparser.add_argument('--beta_start', type=float, default=0.0001, help='beta start')
    argparser.add_argument('--beta_end', type=float, default=0.02, help='beta end')
    argparser.add_argument('--sample_and_save_freq', type=int, default=10, help='sample and save frequency')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['textile','toothbrush','bottle','mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--ddpm', type=float, default=1.0, help='ddim sampling is 0.0, pure ddpm is 1.0')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['textile','toothbrush','bottle','mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--loss_type', type=str, default='huber', help='loss type', choices=['huber','l2', 'l1'])
    argparser.add_argument('--sample_timesteps', type=int, default=300, help='number of timesteps')
    args = argparser.parse_args()
    args.channel_scale_factors = tuple(args.channel_scale_factors)

    return args

def parse_args_CDDPM():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample model')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--timesteps', type=int, default=500, help='number of timesteps')
    argparser.add_argument('--beta_start', type=float, default=0.0001, help='beta start')
    argparser.add_argument('--beta_end', type=float, default=0.02, help='beta end')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['textile','toothbrush','bottle','mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--ddpm', type=float, default=1.0, help='ddim sampling is 0.0, pure ddpm is 1.0')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['textile','toothbrush','bottle','mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--sample_timesteps', type=int, default=500, help='number of timesteps')
    argparser.add_argument('--n_features', type=int, default=128, help='number of features')
    argparser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    argparser.add_argument('--sample_and_save_freq', type=int, default=10, help='sample and save frequency')
    argparser.add_argument('--drop_prob', type=float, default=0.1, help='dropout probability')
    argparser.add_argument('--guide_w', type=float, default=0.5, help='guide weight')
    argparser.add_argument('--ws_test', type = float, nargs='+', default = [0.0, 0.5, 2.0], help='guidance weights for test')
    return argparser.parse_args()

def parse_args_DiffAE():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--manipulate', action='store_true', default=False, help='manipulate latents')
    argparser.add_argument('--batch_size', type=int, default=16, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--timesteps', type=int, default=1000, help='number of timesteps')
    argparser.add_argument('--sample_timesteps', type=int, default=100, help='number of timesteps')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['textile','toothbrush','bottle','mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--embedding_dim', type=int, default=512, help='embedding dimension')
    argparser.add_argument('--model_channels', type=int, nargs='+', default=[64, 128, 256], help='model channels')
    argparser.add_argument('--attention_levels', type=bool, nargs='+', default=[False, True, True], help='attention levels (must match len of model_channels)')
    argparser.add_argument('--num_res_blocks', type=int, default=1, help='number of res blocks')
    argparser.add_argument('--sample_and_save_freq', type=int, default=10, help='sample and save frequency')
    args = argparser.parse_args()
    args.model_channels = tuple(args.model_channels)
    args.attention_levels = tuple(args.attention_levels)
    return args

def parse_args_CycleGAN():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--test', action='store_true', default=False, help='test model')
    argparser.add_argument('--batch_size', type=int, default=1, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=200, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argparser.add_argument('--decay', type=float, default=100, help='epoch to start linearly decaying the learning rate to 0')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--dataset', type=str, default='horse2zebra', help='dataset name', choices=['horse2zebra', 'apple2orange', 'summer2winter_yosemite', 'monet2photo', 'cezanne2photo', 'ukiyoe2photo', 'vangogh2photo', 'maps', 'cityscapes', 'facades', 'iphone2dslr_flower'])
    argparser.add_argument('--checkpoint_A', type=str, default=None, help='checkpoint A path')
    argparser.add_argument('--checkpoint_B', type=str, default=None, help='checkpoint B path')
    argparser.add_argument('--input_size', type=int, default=128, help='input size')
    argparser.add_argument('--in_channels', type=int, default=3, help='in channels')
    argparser.add_argument('--out_channels', type=int, default=3, help='out channels')
    return argparser.parse_args()

def parse_args_CondGAN():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample from model')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argparser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    argparser.add_argument('--beta2', type=float, default=0.999, help='beta2')
    argparser.add_argument('--latent_dim', type=int, default=100, help='latent dimension')
    argparser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    argparser.add_argument('--img_size', type=int, default=32, help='image size')
    argparser.add_argument('--channels', type=int, default=1, help='channels')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample interval')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--n_samples', type=int, default=9, help='number of samples')
    argparser.add_argument('--d', type=int, default=128, help='d')
    return argparser.parse_args()

def parse_args_VanillaGAN():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample from model')
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argparser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    argparser.add_argument('--beta2', type=float, default=0.999, help='beta2')
    argparser.add_argument('--latent_dim', type=int, default=100, help='latent dimension')
    argparser.add_argument('--img_size', type=int, default=32, help='image size')
    argparser.add_argument('--channels', type=int, default=1, help='channels')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample interval')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--n_samples', type=int, default=9, help='number of samples')
    argparser.add_argument('--d', type=int, default=128, help='d')
    return argparser.parse_args()
# EOF
