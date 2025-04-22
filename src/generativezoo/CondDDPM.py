import torch
from data.Dataloaders import *
from models.DDPM.ConditionalDDPM import *
from utils.util import get_args_CDDPM
import wandb

def run(args):
    args.channel_mult = tuple(args.channel_mult)
    args.attention_resolutions = tuple(args.attention_resolutions)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize = True

    if args.train:
        train_dataloader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize, size=args.size, num_workers=args.num_workers)
        model = ConditionalDDPM(in_channels=channels, input_size=input_size, args=args)
        model.train_model(train_dataloader)

    elif args.sample:
        _, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize, size=args.size)
        model = ConditionalDDPM(in_channels=channels, input_size=input_size, args=args)
        if args.checkpoint is not None:
            model.model.load_state_dict(torch.load(args.checkpoint, weights_only=False))
        model.sample()

    elif args.fid:
        _, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize, size=args.size)
        model = ConditionalDDPM(in_channels=channels, input_size=input_size, args=args)
        if args.checkpoint is not None:
            model.model.load_state_dict(torch.load(args.checkpoint, weights_only=False))
        model.fid_sample()

    else:
        raise ValueError('Please specify at least one of the following: train, sample')


if __name__ == "__main__":
    args = get_args_CDDPM().parse_args()
    run(args)
