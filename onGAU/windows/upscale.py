from callbacks import *
from .utils import *
import config

import dearpygui.dearpygui as dpg


upscaler = None


def upscale_image_callback():
    """Callback to upscale the currently shown image."""
    global upscaler

    # Check if it has been defined yet.
    if not config.ESRGAN_MODEL:
        logger.error(
            "ESRGAN model path has not been defined in config.py yet. Link to download the model: https://github.com/xinntao/Real-ESRGAN"
        )
        return

    # Initialize the ESRGAN model if it hasn't been initialized yet.
    if not upscaler:
        logger.info("Initializing ESRGAN model...")
        model_path = utils.append_dir_if_startswith(
            config.ESRGAN_MODEL, FILE_DIR, "models/"
        )

        try:
            upscaler = ESRGAN(model_path, config.DEVICE)
        except ValueError as e:  # When model type cannot be determined.
            status(str(e), logger.error)
            return

    logger.info("Upscaling image...")
    update_window_title("Upscaling image...")
    dpg.set_item_label("upscale_button", "Upscaling image...")

    def callback(upscaled: ESRGANUpscaledImage, error: bool = False):
        dpg.set_item_label("upscale_button", "Upscale Image")
        update_window_title()

        if error:
            logger.error(
                "Too much memory allocated. Enable tiling to reduce memory usage (not implemented yet)."
            )
            return

        logger.success("Finished upscaling.")

        texture_manager.update(upscaled)
        update_image_widget(*texture_manager.current())

    # Upscaling causes first step to take extra long.
    # Putting it in a new thread seems to make it plateau at around 11-12s
    pipelines.esrgan(
        upscaler,
        texture_manager.current()[1],
        dpg.get_value("upscale_amount"),
        callback,
    )


def toggle():
    toggle_item("upscale_window")


def init():
    with dpg.window(label="Upscale", show=False, tag="upscale_window", pos=CENTER):
        dpg.add_input_int(
            label="Upscale Amount",
            default_value=int(user_settings["upscale_amount"]),
            min_value=1,
            width=config.ITEM_WIDTH * 0.75,
            tag="upscale_amount",
        )
        add_tooltip("The amount to upscale the image.")

        dpg.add_button(
            label="Upscale Image",
            tag="upscale_button",
            callback=upscale_image_callback,
        )
