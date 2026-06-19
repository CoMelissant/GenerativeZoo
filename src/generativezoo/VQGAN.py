from models.GAN.VQGAN import VQModel
from utils.util import get_args_VQGAN
from data.Dataloaders import *


def run(args):
    if args.train:
        train_loader, input_size, channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, num_workers=args.num_workers, size=args.size)
        val_loader, _, _ = pick_dataset(args.dataset, mode='val', batch_size = args.batch_size, normalize=True, num_workers=args.num_workers, size=args.size)
        model = VQModel(args, channels, input_size)
        model.train_model(train_loader, val_loader)

    elif args.reconstruct:
        val_loader, input_size, channels = pick_dataset(args.dataset, mode='val', batch_size = args.batch_size, normalize=True, num_workers=args.num_workers, size=args.size)
        model = VQModel(args, channels, input_size)
        model.load_checkpoint(args.checkpoint)
        model.reconstruct(val_loader)

if __name__ == "__main__":
    run(get_args_VQGAN().parse_args())
