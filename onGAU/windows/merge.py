from imagen.utils import InterpolationFuncs, merge
from callbacks import *
from .utils import *
import config

import dearpygui.dearpygui as dpg


def toggle():
    """Callback to toggle the visibility of the checkpoint merger window."""
    dpg.set_value("merge_path_input", os.path.dirname(imagen.model_path))
    dpg.set_value("model1_input", imagen.model_path)
    toggle_item("merge_window")


INTERP_FUNC_MAPPING = {
    "Weighted Sum": InterpolationFuncs.weighted_sum,
    "Add Difference": InterpolationFuncs.add_diff,
    "Sigmoid": InterpolationFuncs.sigmoid,
    "Inverse Sigmoid": InterpolationFuncs.inv_sigmoid,
}


def merge_checkpoint_callback():
    """Callback to merge models."""
    global imagen

    model1, model2, model3 = dpg.get_values(
        ["model1_input", "model2_input", "model3_input"]
    )

    # Get interpolation function from text value.
    interp_func = INTERP_FUNC_MAPPING[dpg.get_value("interp_method_input")]
    path = dpg.get_value("merge_path_input")
    ignore_te = dpg.get_value("ignore_te")
    alpha = dpg.get_value("alpha")

    if model1 != imagen.model_path and load_model(model1):
        return

    dpg.set_item_label("merge_button", "Merging...")
    logger.info(f"Merging models...")

    try:
        # TODO: fix performance issue
        merge(
            alpha,
            interp_func,
            imagen,
            model2,
            model3,
            ignore_te,
        )
    except FileNotFoundError as e:
        status(str(e), logger.error)
        dpg.set_item_label("merge_button", "Merge")
        return

    if os.path.exists(path):
        while True:
            save_option = input(f"{path} already exists. Overwrite (y/n): ")
            if save_option.lower() in ["y", "yes"]:
                break

            path = input("New path: ")
            if not os.path.exists(path):
                break
    else:
        os.mkdir(path)

    imagen.save_weights(path)
    imagen.model_path = path
    dpg.set_value("model", str(path))

    dpg.set_item_label("merge_button", "Merge")
    logger.success("Successfully merged models.")


def init():
    with dpg.window(label="Model Merger", show=False, tag="merge_window", pos=CENTER):
        dpg.add_input_text(
            label="Merge Path", width=config.ITEM_WIDTH, tag="merge_path_input"
        )
        add_tooltip("Output path to dump merged model.")

        dpg.add_input_text(label="Model 1", tag="model1_input", width=config.ITEM_WIDTH)
        add_tooltip("Model 1 path.")

        dpg.add_input_text(label="Model 2", tag="model2_input", width=config.ITEM_WIDTH)
        add_tooltip("Model 2 path.")

        dpg.add_input_text(label="Model 3", tag="model3_input", width=config.ITEM_WIDTH)
        add_tooltip('Model 3 path (only for "Add Difference" interpolation method).')

        dpg.add_combo(
            items=[
                "Weighted Sum",
                "Add Difference",
                "Sigmoid",
                "Inverse Sigmoid",
            ],
            default_value="Weighted Sum",
            label="Interpolation Method",
            tag="interp_method_input",
            width=config.ITEM_WIDTH,
        )
        add_tooltip("The interpolation method to use to merge the models.")

        dpg.add_slider_float(
            label="Alpha",
            max_value=1.00,
            min_value=0.00,
            default_value=0.80,
            clamped=True,
            width=config.ITEM_WIDTH,
            format="%.2f",
            tag="alpha",
        )
        add_tooltip("The ratio to merge the models. 0 makes it the base model.")

        dpg.add_checkbox(
            label="Ignore Text Encoder", default_value=True, tag="ignore_te"
        )
        add_tooltip("If enabled, the text encoder wouldn't been merged.")

        dpg.add_button(
            label="Merge",
            width=config.ITEM_WIDTH / 2,
            callback=merge_checkpoint_callback,
            tag="merge_button",
        )
