"""GenerativeZoo central command line interface.

For models where the input arguments are split over the actions use:
    python -m generativezoo MODEL ACTION arguments

For models with all input arguments on the same level:
    python -m generativezoo MODEL arguments

    or when in the generativezoo sub-folder:
    python -m MODEL arguments.
"""

import importlib

import utils.cli

# Build the central Generative Zoo command line interface and parse the arguments.
central_parser = utils.cli.build_parser()
args, unmatched = central_parser.parse_known_args()


# Get the run function of the selected model and run it with the selected inputs.
if getattr(args, "fcn", None):
    module_name, _, func_name = args.fcn.rpartition(".")
    mod = importlib.import_module(module_name)
    getattr(mod, func_name)(args)
else:
    raise RuntimeError(f"Unexpected GenerativeZoo inputs: {unmatched}")
