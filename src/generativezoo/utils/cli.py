"""GenerativeZoo central command line interface.

Note that the Models in the zoo also define their own CLI.


[For devs] Adding custom dataset command line arguments.

- Register the "custom_dataset" type with the parser and ensure it prepends the specified value with
  "custom_module_".

  parser.register("type", "custom_dataset", lambda s: "custom_module_" + s)

- As the argument use the "custom_dataset" type and set the `dest` to normal dataset argument name.

  add_argument('--dataset', type=str, default=[DATASETS[0]], help='dataset name', choices=DATASETS),
  add_argument('--custom_dataset', type="custom_dataset", help='(Full) name of the custom Dataset loader module to use.', dest="dataset"),

The `data.Dataloaders.pick_dataset' function uses the "custom_module_" prepended to the custom
dataset input as a flag to import the module specified with the text that comes after it. Refer to
the data.Dataloaders module for more requirements on the custom data loader module.
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

    model_parsers = parser.add_subparsers(title="Available models", required=True)

    for model in ModelInterface.get_subclasses():
        # Add the CLI commands for the registered models.
        if model.sub_parser:
            model_parsers.add_parser(model.module, parents=[model.sub_parser], add_help=False)
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
    inputs: Union[List[Tuple[str, List[str], dict]], None] = None
    run_function: str
    sub_parser: Union[argparse.ArgumentParser, None] = None
    model_type: Union[str, None] = None

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
                    subparser.register("type", "custom_dataset", lambda s: "custom_module_" + s)

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
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--from_ui",
            default=False,
            help="Running Models from a GUI. Block results being shown in the backend. Defaults to False.",
        )
        return cls.get_parser(parser).parse_args()

    @classmethod
    def get_subclasses(cls):
        """Get all ModelInterface subclasses on the path."""
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


class ControlNet(ModelInterface):
    module = "Text2Img_ControlNet"
    run_function = "Text2Img_ControlNet.run"
    model_type = "Stable Diffusion"
    inputs = [
        ("--cnet_model_path", [], dict(type=str, help="Path to ControlNet model")),
        ("--cond_image_path", [], dict(type=str, help="Path to conditioning image")),
        ("prompt", [], dict(type=str, help="Image generation prompt.")),
    ]  # fmt: skip


class HierarchicalVAE(ModelInterface):
    module = "HVAE"
    run_function = "HVAE.run"
    model_type = "VAE"

    # Register the input arguments for the model.
    # Argument name, list of actions they are used in, argument parser inputs.
    inputs = [
        ('--dataset',    ["t", "s"],    dict(type=str, default=[DATASETS[0]], help='dataset name', choices=DATASETS)),
        ('--custom_dataset', ["t", "s"], dict(type="custom_dataset", help='(Full) name of the custom dataset loader module to use.', dest="dataset")),
        ('--batch_size', ["t", "s"],    dict(type=int,   default=256, help='batch size')),
        ('--n_epochs',   ["t"],         dict(type=int,   default=100, help='number of epochs')),
        ('--lr',         ["t"],         dict(type=float, default=0.01, help='learning rate')),
        ('--latent_dim', ["t", "s"],    dict(type=int,   default=512, help='latent dimension')),
        ('--checkpoint', ["s"],         dict(type=str,   default=None, help='checkpoint path')),
        # ('--sample_and_save_freq', [],  dict(type=int, default=5, help='sample and save frequency')),
        ('--no_wandb',   ["t"],         dict(action='store_true', default=False, help='disable wandb logging')),
        ('--num_workers', ["t", "s"],   dict(type=int,   default=0, help='number of workers for dataloader'))
    ]  # fmt: skip


class InstructPix2Pix(ModelInterface):
    module = "InstructPix2Pix"
    run_function = "InstructPix2Pix.run"
    model_type = "Stable Diffusion"

    inputs = [
        ("--pix2pix_model", [], dict(type=str, help="The name of the Pix2Pix model to use")),
        ("--image_path", [], dict(type=str, help="The path to the image to edit")),
        ("prompt", [], dict(type=str, help="Image generation prompt.")),
    ]  # fmt: skip


class RealNVP(ModelInterface):
    module = "RNVP"
    run_function = "RNVP.run"
    model_type = "Normalizing Flows"

    inputs = [
        ('--dataset', ["t", "s", "o"], dict(type=str, default=DATASETS[0], help='dataset name', choices=DATASETS)),
        ('--custom_dataset', ["t", "s", "o"], dict(type="custom_dataset", help='(Full) name of the custom dataset loader module to use.', dest="dataset")),
        ('--out_dataset', ["o"], dict(type=str, default='fashionmnist', help='outlier dataset name', choices=DATASETS)),
        ('--custom_out_dataset', ["o"], dict(type="custom_dataset", help='(Full) name of the custom outlier dataset loader module to use.', dest="out_dataset")),
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


class StableDiffusionLoRa(ModelInterface):
    module = "Text2Img_Lora"
    run_function = "Text2Img_Lora.run"
    model_type = "Stable Diffusion"

    inputs = [
        ("--lora_model_path", [], dict(type=str, help="Path to LoRA model")),
        ("prompt", [], dict(type=str, help="Image generation prompt.")),
    ]  # fmt: skip


class AdversarialVAE(ModelInterface):
    module = "AdvVAE"
    run_function = "AdvVAE.run"
    model_type = "GAN"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_AdversarialVAE()


class ConditionalDDPM(ModelInterface):
    module = "CondDDPM"
    run_function = "CondDDPM.run"
    model_type = "DDPM"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_CDDPM()


class CondFlowMatching(ModelInterface):
    module = "CondFM"
    run_function = "CondFM.run"
    model_type = "Flow Matching"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_CondFlowMatching()


class ConditionalGAN(ModelInterface):
    module = "CondGan"
    run_function = "CondGan.run"
    model_type = "GAN"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_CondGAN()


class ConditionalVAE(ModelInterface):
    module = "CondVAE"
    run_function = "CondVAE.run"
    model_type = "VAE"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_ConditionalVAE()


class CycleGAN(ModelInterface):
    module = "CycGAN"
    run_function = "CycGAN.run"
    model_type = "GAN"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_CycleGAN()


class DiffAE(ModelInterface):
    module = "DAE"
    run_function = "DAE.run"
    model_type = "DDPM"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_DiffAE()


class DCGAN(ModelInterface):
    module = "DCGAN"
    run_function = "DCGAN.run"
    model_type = "GAN"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_DCGAN()


class DDPM(ModelInterface):
    module = "DDPM"
    run_function = "DDPM.run"
    model_type = "DDPM"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_DDPM()


class FlowPlusPlus(ModelInterface):
    module = "FlowPP"
    run_function = "FlowPP.run"
    model_type = "Normalizing Flows"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_FlowPP()


class FlowMatching(ModelInterface):
    module = "FM"
    run_function = "FM.run"
    model_type = "Flow Matching"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_FlowMatching()


class Glow(ModelInterface):
    module = "GLOW"
    run_function = "GLOW.run"
    model_type = "Normalizing Flows"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_Glow()


class NCSNv2(ModelInterface):
    module = "NCSNv2"
    run_function = "NCSNv2.run"
    model_type = "Score Matching"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_NCSNv2()


class PixelCNN(ModelInterface):
    module = "P-CNN"
    run_function = "P-CNN.run"
    model_type = "Autoregressive"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_PixelCNN()


class PresGAN(ModelInterface):
    module = "PresGAN"
    run_function = "PresGAN.run"
    model_type = "GAN"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_PresGAN()


class RF(ModelInterface):
    module = "RF"
    run_function = "RF.run"
    model_type = "Flow Matching"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_RectifiedFlows()


class SGM(ModelInterface):
    module = "SGM"
    run_function = "SGM.run"
    model_type = "Score Matching"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_SGM()


class VanillaFlow(ModelInterface):
    module = "VanFlow"
    run_function = "VanFlow.run"
    model_type = "Normalizing Flows"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_VanillaFlow()


class VanillaVAE(ModelInterface):
    module = "VanVAE"
    run_function = "VanVAE.run"
    model_type = "VAE"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_VanillaVAE()


class VQGANTransformer(ModelInterface):
    module = "VQGAN_T"
    run_function = "VQGAN_T.run"
    model_type = "Autoregressive"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_VQGAN_Transformer()


class VQVAETransformer(ModelInterface):
    module = "VQVAE_T"
    run_function = "VQVAE_T.run"
    model_type = "Autoregressive"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_VQVAE_Transformer()


class WGAN(ModelInterface):
    module = "WGAN"
    run_function = "WGAN.run"
    model_type = "GAN"

    # Get the standard ArgumentParser object.
    sub_parser = gzutil.get_args_WassersteinGAN()
