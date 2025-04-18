import argparse
from collections import namedtuple
import importlib
import itertools
import os
from typing import Dict, List, Tuple

from simian.gui import Form, utils, component, composed_component, form

from generativezoo.utils import util as gzutil
import generativezoo.utils.cli as gzcli

REGISTERED_MODELS = {
    model.__name__: (model, model.run_function) for model in gzcli.ModelInterface.get_subclasses()
}


if __name__ == "__main__":
    # Run the zoo_ui app.
    from simian.local import Uiformio

    Uiformio("zoo_ui", window_title="Generative Zoo", debug=True, show_refresh=True)


def gui_init(meta_data: dict) -> dict:
    """Initialize the zoo_ui app."""
    # Register component initialization functions.
    Form.componentInitializer(
        modelOptions=_fill_model_list,
        settings=composed_component.PropertyEditor.get_initializer(column_label="Inputs"),
    )

    # Create the form and load the json builder into it.
    form_obj = form.Form(from_file=__file__)

    return {
        "form": form_obj,
        "navbar": {
            "title": "Simian demo",
            "logo": utils.encodeImage(
                os.path.join(os.path.dirname(__file__), "logo_tasti_light.png")
            ),
        },
    }


def gui_event(meta_data: dict, payload: dict) -> dict:
    # Register our event callbacks to the events.
    Form.eventHandler(
        modelSelectionChanged=model_change,
        actionSelectionChanged=action_change,
        runModel=run_model,
    )

    # Execute the callback.
    callback = utils.getEventFunction(meta_data, payload)
    return callback(meta_data, payload)


def run_model(meta_data: dict, payload: dict) -> dict:
    """Run the registered model with the settings in the editor."""
    selected_model, _ = utils.getSubmissionData(payload, "selectModel")
    selected_action, _ = utils.getSubmissionData(payload, "selectAction")

    if len(selected_model) == 0:
        # No model selected.
        pass
    else:
        # Get the properties of the registered models.
        model_props = REGISTERED_MODELS[selected_model["value"]]
        model = model_props[0]

        # Get the argument names from the CLI.
        _, argument_names = properties_from_interface(model, selected_action)

        # Get the selected values from the setting editor.
        table_values, _ = utils.getSubmissionData(payload, key="settings")
        value_list = composed_component.PropertyEditor.get_values(table_values)

        if model.inputs:
            # Ensure action names are in the inputs and that the selected one is set to True.
            action_names = [action[0] for action in model.actions.values()]
            action_values = [name == selected_action for name in action_names]

            argument_names += action_names
            value_list += action_values

        argument_names += ["from_ui"]
        value_list += [True]

        # Create a namedtuple that will contain the model input arguments.
        input_args = namedtuple("args", argument_names)

        # Derive the module and function name from the the full function name.
        module_name, _, run_func = model_props[1].rpartition(".")
        module = importlib.import_module(module_name)
        model_run_function = getattr(module, run_func)

        # Execute the model run function with the inputs from the ui.
        model_run_function(input_args(**{k: v for k, v in zip(argument_names, value_list)}))

    return payload


def action_change(meta_data: dict, payload: dict) -> dict:
    _change_options(payload)
    return payload


def model_change(meta_data: dict, payload: dict) -> dict:
    selected_model, _ = utils.getSubmissionData(payload, "selectModel")
    action_aliases = []

    if selected_model:
        props = REGISTERED_MODELS[selected_model["value"]]
        model = props[0]

        if model.inputs:
            action_aliases = set(itertools.chain(*[line[1] for line in model.inputs]))
        else:
            pass

    action_names = [model.actions[alias][0] for alias in action_aliases]
    utils.setSubmissionData(payload, "actionOptions", action_names)
    _change_options(payload)
    return payload


def _change_options(payload: dict) -> dict:
    """Model selection changed, update the available settings in the editor."""
    selected_model, _ = utils.getSubmissionData(payload, "selectModel")
    selected_action, _ = utils.getSubmissionData(payload, "selectAction")

    if len(selected_model) == 0:
        new_settings = [{}]
    else:
        # Get the properties of the selected model.
        props = REGISTERED_MODELS[selected_model["value"]]

        # Convert the Argument parser properties to a PropertyEditor input.
        settings_list, _ = properties_from_interface(props[0], selected_action)
        new_settings = composed_component.PropertyEditor.prepare_values(settings_list)

    # Update the contents of the settings editor.
    utils.setSubmissionData(payload, "settings", new_settings)


def _fill_model_list(comp):
    """Fill the model list with the registered models."""
    comp.defaultValue = [{"label": m, "value": m} for m in REGISTERED_MODELS.keys()]


def properties_from_interface(model, arg: str = "train"):
    settings = []
    argument_names = []
    action_alias = [k for k, vals in model.actions.items() if vals[0] == arg]

    if model.inputs:
        # Model definition uses lists of inputs.
        for line in model.inputs:
            if len(action_alias) == 0 or action_alias[0] not in line[1]:
                continue

            required = (len(line[0]) == 1 and not line[0].startswith("-")) or line[2].get(
                "required", False
            )
            name = line[0].lstrip("-")

            _process_args(name, required, line[2], settings, argument_names)

    else:
        # Model definition contains an ArgumentParser object. (Legacy)
        for arg in model.sub_parser._actions:
            if isinstance(arg, argparse._HelpAction):
                # Standard CLI help, no input for user to specify.
                pass
            else:
                required = (
                    len(arg.option_strings) == 1 and not arg.option_strings[0].startswith("-")
                ) or arg.required
                name = arg.dest

                _process_args(name, required, arg.__dict__, settings, argument_names)

    return settings, argument_names


def _process_args(
    name: str, required: bool, meta: dict, settings: list, argument_names: list
) -> None:
    """Process CLI argument meta data to a PropertyEditor row."""
    type_str, extra_options = _determine_type(meta)

    settings.append(
        {
            "datatype": type_str,
            "label": name,
            "tooltip": meta.get("help", None),
            "required": required,
            "defaultValue": meta.get("default", None),
        }
        | extra_options
    )

    argument_names.append(name)


def _determine_type(meta: dict) -> Tuple[str, dict]:
    """Determine the datatype that corresponds with the cli argument meta data."""
    extra_options = {}
    type_str = None

    data_type = meta.get("type", None)
    if choices := meta.get("choices", None):
        type_str = "select"
        extra_options = extra_options | {"allowed": choices}
    elif data_type is None and meta.get("default", None) is not None:
        # Determine via the default value. (May be `False`)
        data_type = type(meta["default"])

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
