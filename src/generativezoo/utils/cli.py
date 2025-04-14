"""GenerativeZoo central command line interface.

Note that the Models in the zoo also define their own CLI.
"""

import argparse
from typing import List, Tuple, Union

import utils.util as gzutil


DATASETS = ['mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'tinyimagenet','imagenet', 'celeba']  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    """Create the Generative Zoo central command line interface contents.

    Used by generativezoo.__main__ to get the CLI.
    """
    parser = argparse.ArgumentParser(prog="GenerativeZoo")
    parser.add_argument(
        "--from_ui",
        default=False,
        help="Running Models from a GUI. Block results being shown in the backend. Defaults to False.",
    )

    model_parsers = parser.add_subparsers(title="Available models:", required=True)

    for model in ModelInterface.get_subclasses():
        # Add the CLI commands for the registered models.
        if model.sub_parser:
            model_parsers.add_parser(model.module, parents=[model.sub_parser], add_help=False)
            # model.sub_parser
            # o = model_parsers.add_parser("a")
        else:
            model_parser = model_parsers.add_parser(model.module)
            model.get_parser(model_parser)

    return parser


class ModelInterface:
    """Model registration super class.

    Two modes:
    - sub_parser with actions and inputs defined in the `inputs` list.
    - sub_parser created from a standalone ArgumentParser object in the `sub_parser` attribute.
    """

    # Registered model actions; flag, name, Description.
    actions = {
        "t": ("train", "Train model"),
        "s": ("sample", "Sample model"),
        "o": ("outlier_detection", "Outlier detection"),
    }
    module: str  # Importing may be expensive, so register as string.

    # Model inputs registration:
    inputs: Union[List[Tuple[str, List[str], dict]], None]
    run_function: str
    sub_parser: Union[argparse.ArgumentParser, None] = None

    @classmethod
    def get_parser(cls, parser: argparse.ArgumentParser):
        """Create an arguments parser from the class' inputs property.

        Split / duplicate input arguments over the registered actions.
        """
        subparsers = parser.add_subparsers(title="Available model actions:", required=True)
        sub_dict = {}
        actions_off = {v[0]: False for k, v in cls.actions.items()}

        for line in cls.inputs:
            for act in line[1]:
                if act in sub_dict:
                    # A subparser for this action was already created. Reuse it.
                    subparser = sub_dict[act]
                elif act in cls.actions:
                    # Create a sub parser that automatically adds itself to the arguments.
                    subparser = subparsers.add_parser(cls.actions[act][0], help=cls.actions[act][1])

                    # The models expect fields for all actions in the CLI output.
                    # Ensure the current args.action field is set to True and the rest remains False
                    action_defaults = actions_off | {cls.actions[act][0]: True}
                    subparser.set_defaults(**action_defaults)
                    subparser.set_defaults(fcn=cls.run_function)
                    sub_dict.update({act: subparser})
                else:
                    raise RuntimeError("Unknown action alias %s encountered", act)

                # Add a subparser with the argument name and the dict with meta data.
                subparser.add_argument(line[0], **line[2])

        return parser

    def get_args(cls) -> argparse.Namespace:
        return cls.get_parser().parse_args()

    @classmethod
    def get_subclasses(cls):
        """Get all ModelInterface subclasses on the path."""
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


class HierarchicalVAE(ModelInterface):
    module = "HVAE"
    run_function = "HVAE.run"

    # Register the input arguments for the model.
    # Argument name, list of actions they are used in, argument parser inputs.
    inputs = [
        ('--dataset',    ["t", "s"],    dict(type=str,   default=DATASETS[0], help='dataset name', choices=DATASETS)),
        ('--batch_size', ["t", "s"],    dict(type=int,   default=256, help='batch size')),
        ('--n_epochs',   ["t"],         dict(type=int,   default=100, help='number of epochs')),
        ('--lr',         ["t"],         dict(type=float, default=0.01, help='learning rate')),
        ('--latent_dim', ["t", "s"],    dict(type=int,   default=512, help='latent dimension')),
        ('--checkpoint', ["s"],         dict(type=str,   default=None, help='checkpoint path')),
        # ('--sample_and_save_freq', [],  dict(type=int, default=5, help='sample and save frequency')),
        ('--no_wandb',   ["t"],         dict(action='store_true', default=False, help='disable wandb logging')),
        ('--num_workers', ["t", "s"],   dict(type=int,   default=0, help='number of workers for dataloader'))
    ]  # fmt: skip


class RealNVP(ModelInterface):
    module = "RNVP"
    run_function = "RNVP.run"

    inputs = [
        ('--dataset', ["t", "s", "o"], dict(type=str, default=DATASETS[0], help='dataset name', choices=DATASETS)),
        ('--out_dataset', ["o"], dict(type=str, default='fashionmnist', help='outlier dataset name', choices=DATASETS)),
        ('--batch_size', ["t", "o"], dict(type=int, default=128, help='batch size')),
        ('--n_epochs', ["t"], dict(type=int, default=100, help='number of epochs')),
        ('--lr', ["t"], dict(type=float, default=1e-3, help='learning rate')),
        # ('--weight_decay', ["t", "o"], dict(type=float, default=1e-5, help='weight decay')),
        ('--max_grad_norm', ["t"], dict(type=float, default=100.0, help='max grad norm')),
        ('--sample_and_save_freq', ["t"], dict(type=int, default=5, help='sample and save frequency')),
        ('--num_scales', ["t", "s", "o"], dict(type=int, default=2, help='number of scales')),
        ('--mid_channels', ["t", "s", "o"], dict(type=int, default=64, help='mid channels')),
        ('--num_blocks', ["t", "s", "o"], dict(type=int, default=8, help='number of blocks')),
        ('--checkpoint', ["s", "o"], dict(type=str, default=None, help='checkpoint path')),
        ('--no_wandb', ["t"], dict(action='store_true', default=False, help='disable wandb logging')),
        ('--num_workers', ["t"], dict(type=int, default=0, help='number of workers for dataloader')),
    ]  # fmt: skip


class PixelCNN(ModelInterface):
    module = "P-CNN"
    run_function = "P-CNN.run"
    inputs = None

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_PixelCNN()
