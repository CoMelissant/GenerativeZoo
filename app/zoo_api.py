import argparse
from collections import namedtuple
import importlib
import os
import json

from typing import Dict, List, Tuple

from utils import util as gzutil

from flask import Flask, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "clip_input")
app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



REGISTERED_MODELS = {
    # Model name: GET argument parser function, full run function name.
    "HierarchicalVAE": (gzutil.get_args_HierarchicalVAE, "generativezoo.HVAE.run_HVAE"),
    "RealNVP": (gzutil.get_args_RealNVP, "generativezoo.RNVP.run_RealNVP"),
}



@app.route('/')
def get_index():
    return "GenerativeZoo running"


@app.route('/get_model_props/<model>', methods=(["POST","GET"]))
def get_model_props(model):

    
    print(model)
    
    # Get the properties of the selected model.
    props = REGISTERED_MODELS[model]

    # Convert the Argument parser properties to a PropertyEditor input.
    settings_list, argument_names = properties_from_parser(props[0]())


    return jsonify({"names": argument_names, "settings": settings_list})


# def run_model(meta_data: dict, payload: dict) -> dict:
#     """Run the registered model with the settings in the editor."""
#     selected_model, _ = utils.getSubmissionData(payload, "selectModel")

#     if len(selected_model) == 0:
#         # No model selected.
#         pass
#     else:
#         # Get the properties of the registered models.
#         props = REGISTERED_MODELS[selected_model["value"]]

#         # Get the argument names from the CLI.
#         _, argument_names = properties_from_parser(props[0]())

#         # Get the selected values from the setting editor.
#         table_values, _ = utils.getSubmissionData(payload, key="settings")
#         value_list = composed_component.PropertyEditor.get_values(table_values)

#         # Add ui notification
#         argument_names.append('from_ui');
#         value_list.append(True);

#         # Create a namedtuple that will contain the model input arguments.
#         input_args = namedtuple("args", argument_names)

#         # Derive the module and function name from the the full function name.
#         moduleName, _, builderName = props[1].rpartition(".")
#         module = importlib.import_module(moduleName)
#         model_run_function = getattr(module, builderName)
        
#         # Execute the model run function with the inputs from the ui.
#         if value_list[argument_names.index('sample')]:
#             outputs = []

#             result = model_run_function(input_args(**{k: v for k, v in zip(argument_names, value_list)}))

#             plot_obj, _ = utils.getSubmissionData(payload, "image")
#             plot_obj.clearFigure()
#             plot_obj.figure = px.imshow(result)
            
#               #  grid.permute(1, 2, 0),
#               #  name="Average last<br>ten years",
#               #  )
            
#             payload, _ = utils.setSubmissionData(payload, "image", plot_obj)

#         else:
#             model_run_function(input_args(**{k: v for k, v in zip(argument_names, value_list)}))
#             outputs = []
        

#     return payload


# def model_change(meta_data: dict, payload: dict) -> dict:
#     """Model selection changed, update the available settings in the editor."""
#     selected_model, _ = utils.getSubmissionData(payload, "selectModel")

#     if len(selected_model) == 0:
#         new_settings = [{}]
#     else:
#         # Get the properties of the selected model.
#         props = REGISTERED_MODELS[selected_model["value"]]

#         # Convert the Argument parser properties to a PropertyEditor input.
#         settings_list, _ = properties_from_parser(props[0]())
#         new_settings = composed_component.PropertyEditor.prepare_values(settings_list)

#     # Update the contents of the settings editor.
#     utils.setSubmissionData(payload, "settings", new_settings)

#     return payload


def _fill_model_list(component):
    """Fill the model list with the registered models."""
    component.defaultValue = [{"label": m, "value": m} for m in REGISTERED_MODELS.keys()]


def properties_from_parser(args_parser) -> Tuple[List[Dict], List[str]]:
    """Get the properties from the argument parser in the inputs.

    Args:
        args_parser: argparse.ArgumentParser object to process.

    Returns:
        settings:   List of dicts with PropertyEditor compliant values from the argument parser
            arguments.
        argument_names: List of input argument names defined in the argument parser.
    """
    # TODO:
    #  - type can be a callable that accepts a string or the name of a registered type.
    #    https://docs.python.org/3/library/argparse.html#type
    settings = []
    argument_names = []

    for action in args_parser._actions:
        if isinstance(action, argparse._HelpAction):
            # Standard CLI help, no input for user to specify.
            pass
        else:
            # names = [n.lstrip("-") for n in action.option_strings]
            # name = max(names, key=len)
            required = (
                len(action.option_strings) == 1 and not action.option_strings[0].startswith("-")
            ) or action.required
            name = action.dest

            type_str, extra_options = _determine_type(action)

            settings.append(
                {
                    "datatype": type_str,
                    "label": name,
                    "tooltip": action.help,
                    "required": required,
                    "defaultValue": action.default,
                }
                | extra_options
            )

            argument_names.append(action.dest)

    return settings, argument_names


def _determine_type(action) -> Tuple[str, dict]:
    """Determine the datatype that corresponds with the cli action."""
    extra_options = {}
    type_str = None

    data_type = action.type
    if action.choices is not None:
        type_str = "select"
        extra_options = extra_options | {"allowed": action.choices}
    elif data_type is None and action.default is not None:
        # Determine via the default value.
        data_type = type(action.default)
    # else:
        # The default data type is str.
        # type_str = "text"

    if type_str is not None:
        # Type is already determined.
        pass
    elif data_type is str:
        type_str = "text"
    elif data_type is bool:
        type_str = "boolean"
    elif data_type is int:
        type_str = "numeric"
        extra_options = extra_options | {"decimalLimit": 0}
    elif data_type is float:
        type_str = "numeric"

    return type_str, extra_options



if __name__ == "__main__":
    # Run the zoo_ui app.

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(port=5005, host='0.0.0.0')