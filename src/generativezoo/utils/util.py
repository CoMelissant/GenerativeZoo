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
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes', 'xray'])
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes'])
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
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes', 'xray', 'tinyimagenet', 'bottle'])
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
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes', 'xray', 'tinyimagenet', 'places365', 'cifar100', 'dtd', 'bottle', 'ssb', 'ninco', 'inaturalist', 'openimageo'])
    return argparser.parse_args()

def parse_args_VanillaSGM():
    argparser = argparse.ArgumentParser()
    # show choices: mnist | cifar10 | fashionmnist | chestmnist | octmnist | tissuemnist | pneumoniamnist | svhn
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes'])
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes'])
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
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['textile','toothbrush','bottle','mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes'])
    argparser.add_argument('--ddpm', type=float, default=1.0, help='ddim sampling is 0.0, pure ddpm is 1.0')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['textile','toothbrush','bottle','mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes'])
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
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['textile','toothbrush','bottle','mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes'])
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
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes', 'xray', 'tinyimagenet'])
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes', 'xray', 'tinyimagenet', 'places365', 'dtd', 'cifar100', 'ssb', 'ninco', 'inaturalist', 'openimageo'])
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lrg', type=float, default=0.0002, help='learning rate generator')
    argparser.add_argument('--lrd', type=float, default=0.0002, help='learning rate discriminator')
    argparser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    argparser.add_argument('--beta2', type=float, default=0.999, help='beta2')
    argparser.add_argument('--latent_dim', type=int, default=100, help='latent dimension')
    argparser.add_argument('--img_size', type=int, default=32, help='image size')
    argparser.add_argument('--channels', type=int, default=1, help='channels')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample interval')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--discriminator_checkpoint', type=str, default=None, help='discriminator checkpoint path')
    argparser.add_argument('--n_samples', type=int, default=9, help='number of samples')
    argparser.add_argument('--d', type=int, default=128, help='d')
    return argparser.parse_args()

def parse_args_WassersteinGAN():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample from model')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes', 'xray', 'tinyimagenet'])
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes', 'xray', 'tinyimagenet', 'ssb'])
    argparser.add_argument('--batch_size', type=int, default=256, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--latent_dim', type=int, default=100, help='latent dimension')
    argparser.add_argument('--d', type=int, default=64, help='d')
    argparser.add_argument('--lrg', type=float, default=0.0002, help='learning rate generator')
    argparser.add_argument('--lrd', type=float, default=0.0002, help='learning rate discriminator')
    argparser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    argparser.add_argument('--beta2', type=float, default=0.999, help='beta2')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample interval')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--discriminator_checkpoint', type=str, default=None, help='discriminator checkpoint path')
    argparser.add_argument('--gp_weight', type=float, default=10.0, help='gradient penalty weight')
    argparser.add_argument('--n_critic', type=int, default=5, help='number of critic updates per generator update')
    argparser.add_argument('--n_samples', type=int, default=9, help='number of samples')

    return argparser.parse_args()

def parse_args_PresGAN():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample from model')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'xray', 'tinyimagenet'])
    ###### Model arguments
    argparser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    argparser.add_argument('--ngf', type=int, default=64)
    argparser.add_argument('--ndf', type=int, default=64)

    ###### Optimization arguments
    argparser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for')
    argparser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    argparser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    argparser.add_argument('--lrE', type=float, default=0.0002, help='learning rate, default=0.0002')
    argparser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')

    ###### Checkpointing and Logging arguments
    argparser.add_argument('--checkpoint', type=str, default=None, help='a given checkpoint file for generator')
    argparser.add_argument('--discriminator_checkpoint', type=str, default=None, help='a given checkpoint file for discriminator')
    argparser.add_argument('--sigma_checkpoint', type=str, default=None, help='a given file for logsigma for the generator')
    argparser.add_argument('--num_gen_images', type=int, default=16, help='number of images to generate for inspection')

    ###### PresGAN-specific arguments
    argparser.add_argument('--sigma_lr', type=float, default=0.0002, help='generator variance')
    argparser.add_argument('--lambda_', type=float, default=0.01, help='entropy coefficient')
    argparser.add_argument('--sigma_min', type=float, default=0.01, help='min value for sigma')
    argparser.add_argument('--sigma_max', type=float, default=0.3, help='max value for sigma')
    argparser.add_argument('--logsigma_init', type=float, default=-1.0, help='initial value for log_sigma_sian')
    argparser.add_argument('--num_samples_posterior', type=int, default=2, help='number of samples from posterior')
    argparser.add_argument('--burn_in', type=int, default=2, help='hmc burn in')
    argparser.add_argument('--leapfrog_steps', type=int, default=5, help='number of leap frog steps for hmc')
    argparser.add_argument('--flag_adapt', type=int, default=1, help='0 or 1')
    argparser.add_argument('--delta', type=float, default=1.0, help='delta for hmc')
    argparser.add_argument('--hmc_learning_rate', type=float, default=0.02, help='lr for hmc')
    argparser.add_argument('--hmc_opt_accept', type=float, default=0.67, help='hmc optimal acceptance rate')
    argparser.add_argument('--stepsize_num', type=float, default=1.0, help='initial value for hmc stepsize')
    argparser.add_argument('--restrict_sigma', type=int, default=0, help='whether to restrict sigma or not')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'xray', 'tinyimagenet', 'places365', 'dtd', 'cifar100', 'ssb', 'ninco', 'inaturalist', 'openimageo'])

    return argparser.parse_args()

def parse_args_Glow():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample from model')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes'])
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes'])
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argparser.add_argument('--hidden_channels', type=int, default=64, help='hidden channels')
    argparser.add_argument('--K', type=int, default=8, help='Number of layers per block')
    argparser.add_argument('--L', type=int, default=3, help='number of blocks')
    argparser.add_argument('--actnorm_scale', type=float, default=1.0, help='act norm scale')
    argparser.add_argument('--flow_permutation', type=str, default='invconv', help='flow permutation', choices=['invconv', 'shuffle', 'reverse'])
    argparser.add_argument('--flow_coupling', type=str, default='affine', help='flow coupling, affine ', choices=['additive', 'affine'])
    argparser.add_argument('--LU_decomposed', action='store_true', default=False, help='Train with LU decomposed 1x1 convs')
    argparser.add_argument('--learn_top', action='store_true', default=False, help='learn top layer (prior)')
    argparser.add_argument('--y_condition', action='store_true', default=False, help='Class Conditioned Glow')
    argparser.add_argument('--y_weight', type=float, default=0.01, help='weight of class condition')
    argparser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--n_bits', type=int, default=8, help='number of bits')
    return argparser.parse_args()

def parse_args_NCSNv2():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample from model')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes'])
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'cityscapes'])
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argparser.add_argument('--nf', type=int, default=128, help='number of filters')
    argparser.add_argument('--act', type=str, default='elu', help='activation', choices=['relu', 'elu', 'swish'])
    argparser.add_argument('--centered', action='store_true', default=False, help='centered')
    argparser.add_argument('--sigma_min', type=float, default=0.01, help='min value for sigma')
    argparser.add_argument('--sigma_max', type=float, default=0.3, help='max value for sigma')
    argparser.add_argument('--num_scales', type=int, default=232, help='number of scales')
    argparser.add_argument('--normalization', type=str, default='InstanceNorm++', help='Normalization', choices=['InstanceNorm', 'GroupNorm', 'VarianceNorm', 'InstanceNorm++'])
    argparser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    argparser.add_argument('--ema_decay', type=float, default=0.999, help='ema decay')
    argparser.add_argument('--continuous', action='store_true', default=False, help='continuous')
    argparser.add_argument('--reduce_mean', action='store_true', default=False, help='reduce mean')
    argparser.add_argument('--likelihood_weighting', action='store_true', default=False, help='likelihood weighting')
    argparser.add_argument('--beta1', type=float, default=0.9, help='beta1')
    argparser.add_argument('--beta2', type=float, default=0.999, help='beta2')
    argparser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    argparser.add_argument('--warmup', type=int, default=0, help='warmup')
    argparser.add_argument('--grad_clip', type=float, default=-1.0, help='grad clip')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--sampler', type=str, default='pc', help='sampler name', choices=['pc', 'ode'])
    argparser.add_argument('--predictor', type=str, default='none', help='predictor', choices=['none', 'em', 'rd', 'as'])
    argparser.add_argument('--corrector', type=str, default='ald', help='corrector', choices=['none', 'l', 'ald'])
    argparser.add_argument('--snr', type=float, default=0.176, help='signal to noise ratio')
    argparser.add_argument('--n_steps', type=int, default=5, help='number of steps')
    argparser.add_argument('--probability_flow', action='store_true', default=False, help='probability flow')
    argparser.add_argument('--noise_removal', action='store_true', default=False, help='noise removal')
    return argparser.parse_args()
# EOF
