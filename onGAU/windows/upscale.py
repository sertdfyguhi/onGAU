from imagen import RealESRGAN, LatentUpscaler
from callbacks import *
from .utils import *
import config

import dearpygui.dearpygui as dpg


upscaler_type = None
upscaler = None

ESRGAN_SHOW_ITEMS = ["upscale_amount"]
ESRGAN_HIDE_ITEMS = [
    "u_step_count",
    "u_guidance_scale",
    "u_attention_slicing",
    "u_vae_slicing",
    "u_xformers_memory_attention",
    "u_model_cpu_offload",
]


def upscaler_type_callback(_, value: str):
    match value:
        case "RealESRGAN":
            for tag in ESRGAN_SHOW_ITEMS:
                dpg.show_item(tag)

            for tag in ESRGAN_HIDE_ITEMS:
                dpg.hide_item(tag)

        case "Latent":
            for tag in ESRGAN_SHOW_ITEMS:
                dpg.hide_item(tag)

            for tag in ESRGAN_HIDE_ITEMS:
                dpg.show_item(tag)


def checkbox_callback(tag: str, value: bool):
    """
    Callback for most upscaler checkbox settings.
    Enables and disables settings based on the tag.
    """
    attr = tag[2:]
    if getattr(upscaler, f"{attr}_enabled", None) == value:
        return

    func_name = f'{"enable_" if value else "disable_"}{attr}'

    try:
        getattr(upscaler, func_name)()
    except Exception as e:
        status(str(e), logger.error)
        dpg.set_value(tag, not value)


def toggle_xformers_callback(_, value: bool):
    """Callback to toggle upscaler xformers."""
    if not torch.cuda.is_available():
        if value:
            dpg.set_value("u_xformers_memory_attention", False)
            status("Xformers is only available for cuda.", logger.error)

        return

    try:
        checkbox_callback("u_xformers_memory_attention", value)
    except ModuleNotFoundError:
        imagen.disable_xformers_memory_attention()
        dpg.set_value("u_xformers_memory_attention", False)
        status(
            "You don't have xformers installed. Please run `pip3 install xformers`.",
            logger.error,
        )


def upscale_image_callback():
    """Callback to upscale the currently shown image."""
    global upscaler, upscaler_type

    upscaler_type_ = dpg.get_value("upscaler_type")

    model = dpg.get_value("upscale_model")
    if not model:
        status("Upscale model path has not been set.", logger.error)
        return

    model_path = utils.append_dir_if_startswith(model, FILE_DIR, "models/")

    # Initialize the ESRGAN model if it hasn't been initialized yet.
    if not upscaler or upscaler_type_ != upscaler_type:
        upscaler_type = upscaler_type_
        logger.info(f"Initializing {upscaler_type} model...")

        if hasattr(upscaler, "_pipeline"):
            del upscaler._pipeline

        try:
            match upscaler_type:
                case "RealESRGAN":
                    try:
                        upscaler = RealESRGAN(model_path, config.DEVICE)
                    except ValueError as e:  # When model type cannot be determined.
                        status(str(e), logger.error)
                        return

                case "Latent":
                    upscaler = LatentUpscaler(model_path, config.DEVICE)
                    upscaler.disable_safety_checker()

                    for tag in ESRGAN_HIDE_ITEMS:
                        dpg.get_item_callback(tag)(tag, dpg.get_value(tag))
        except FileNotFoundError as e:
            status(str(e), logger.error)
            return

    dpg.hide_item("status_text")

    logger.info("Upscaling image...")
    update_window_title("Upscaling image...")
    dpg.set_item_label("upscale_button", "Upscaling image...")

    def callback(upscaled, error):
        dpg.set_item_label("upscale_button", "Upscale Image")
        update_window_title()

        if error:
            status(str(error), logger.error)
            return

        logger.success("Finished upscaling.")

        texture_manager.update(upscaled)
        update_image_widget(*texture_manager.current())

    match upscaler_type:
        case "RealESRGAN":
            # Upscaling causes first step to take extra time.
            # Putting it in a new thread seems to make it plateau at around 11-12s
            pipelines.upscale(
                upscaler,
                texture_manager.current()[1],
                callback,
                upscale=dpg.get_value("upscale_amount"),
            )
        case "Latent":
            pipelines.upscale(
                upscaler,
                # Change to using latents.
                texture_manager.current()[1],
                callback,
            )


def toggle():
    toggle_item("upscale_window")


def init():
    with dpg.window(label="Upscale", show=False, tag="upscale_window", pos=CENTER):
        upscaler_type = user_settings["upscaler_type"]

        dpg.add_combo(
            items=["RealESRGAN", "Latent"],
            label="Upscaler Type",
            default_value=upscaler_type,
            callback=upscaler_type_callback,
            width=config.ITEM_WIDTH,
            tag="upscaler_type",
        )
        add_tooltip("The type of upscaler to use.")

        dpg.add_input_text(
            label="Model",
            default_value=user_settings["upscale_model"],
            width=config.ITEM_WIDTH,
            tag="upscale_model",
        )
        dpg.add_input_int(
            label="Upscale Amount",
            default_value=int(user_settings["upscale_amount"]),
            min_value=1,
            width=config.ITEM_WIDTH,
            tag="upscale_amount",
            show=upscaler_type == "RealESRGAN",
        )
        add_tooltip("The amount to upscale the image.")

        dpg.add_input_int(
            label="Step Count",
            default_value=int(user_settings["u_step_count"]),
            min_value=1,
            width=config.ITEM_WIDTH,
            tag="u_step_count",
            show=upscaler_type == "Latent",
        )
        dpg.add_input_float(
            label="Guidance Scale",
            default_value=float(user_settings["u_guidance_scale"]),
            min_value=0.0,
            max_value=50.0,
            format="%.1f",
            width=config.ITEM_WIDTH,
            tag="u_guidance_scale",
            show=upscaler_type == "Latent",
        )

        dpg.add_checkbox(
            label="Enable Attention Slicing",
            tag="u_attention_slicing",
            default_value=user_settings["u_attention_slicing"] == "True",
            callback=checkbox_callback,
            show=upscaler_type == "Latent",
        )
        add_tooltip(
            "Slices the computation into multiple steps. Increases performance on MPS."
        )

        dpg.add_checkbox(
            label="Enable Vae Slicing",
            tag="u_vae_slicing",
            default_value=user_settings["u_vae_slicing"] == "True",
            callback=checkbox_callback,
            show=upscaler_type == "Latent",
        )
        add_tooltip("VAE decodes one image at a time.")

        dpg.add_checkbox(
            label="Enable xFormers Memory Efficient Attention",
            tag="u_xformers_memory_attention",
            default_value=user_settings["u_xformers_memory_attention"] == "True",
            callback=toggle_xformers_callback,
            show=upscaler_type == "Latent",
        )

        dpg.add_checkbox(
            label="Enable Model CPU Offload",
            tag="u_model_cpu_offload",
            default_value=user_settings["u_model_cpu_offload"] == "True",
            callback=checkbox_callback,
            show=upscaler_type == "Latent",
        )
        add_tooltip(
            "Offloads the model onto the CPU. Reduces memory usage while keeping performance at best."
        )

        dpg.add_button(
            label="Upscale Image",
            tag="upscale_button",
            callback=upscale_image_callback,
        )
