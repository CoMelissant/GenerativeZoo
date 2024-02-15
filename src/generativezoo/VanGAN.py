from models.GAN.VanillaGAN import *
from data.Dataloaders import *
from utils.util import parse_args_VanillaGAN
import torch
import wandb

args = parse_args_VanillaGAN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.train:
    wandb.init(project='VanillaGAN',
               config={
                   'dataset': args.dataset,
                   'batch_size': args.batch_size,
                   'n_epochs': args.n_epochs,
                   'latent_dim': args.latent_dim,
                   'd': args.d,
                   'lr': args.lr,
                   'beta1': args.beta1,
                   'beta2': args.beta2,
                   'sample_and_save_freq': args.sample_and_save_freq
               },
               name = 'VanillaGAN_{}'.format(args.dataset))
    train_dataloader, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=args.batch_size, normalize = True, size = 32)
    model = VanillaGAN(n_epochs = args.n_epochs, device = device, latent_dim = args.latent_dim, d = args.d, channels = channels, lr = args.lr, beta1 = args.beta1, beta2 = args.beta2, img_size = input_size, sample_and_save_freq = args.sample_and_save_freq, dataset=args.dataset)
    model.train_model(train_dataloader)

elif args.sample:
    _, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=1, normalize = True, size = 32)
    model = Generator(latent_dim = args.latent_dim, channels = channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    model.sample(n_samples = args.n_samples, device = device)
else:
    raise Exception('Please specify either --train or --sample')