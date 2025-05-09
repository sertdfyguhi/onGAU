from imagen import Text2Img, Img2Img, GeneratedImage
from settings_manager import SettingsManager
from texture_manager import TextureManager
from theme_manager import ThemeManager
import logger, config, pipelines, utils

from PIL import Image, UnidentifiedImageError
from imagesize import get as get_imsize
import dearpygui.dearpygui as dpg
import torch
import time
import os
import re

dpg.create_context()

# Constants
FONT = os.path.join(os.path.dirname(__file__), "fonts", config.FONT)
GENERATING_MESSAGE = logger.create("Generating... ", [logger.INFO, logger.BOLD])

print(
    logger.create(
        f"Using device {logger.create(config.DEVICE, [logger.BOLD])}.", [logger.INFO]
    )
)


def load_scheduler(scheduler: str):
    """Load a scheduler."""
    logger.info(f"Loading scheduler {scheduler}...")
    imagen.set_scheduler(scheduler)


# load user settings
theme_manager = ThemeManager(config.THEME_DIR)
settings_manager = SettingsManager(config.USER_SETTINGS_FILE, theme_manager)
user_settings = settings_manager.get_settings("main")

imagen_class = Text2Img if user_settings["pipeline"].endswith("Text2Img") else Img2Img
use_LPWSD = user_settings["lpwsd_pipeline"] == "True"
sdxl = user_settings["pipeline"].startswith("SDXL")
model_path = user_settings["model"]


logger.info(f"Loading {model_path}...")

try:
    imagen = imagen_class(
        model_path, config.DEVICE, user_settings["precision"], use_LPWSD, sdxl
    )
except (FileNotFoundError, OSError):
    logger.error(f"{model_path} does not exist, falling back to default model.")

    model_path = config.DEFAULT_MODEL
    logger.info(f"Loading {model_path}...")
    imagen = imagen_class(
        model_path, config.DEVICE, user_settings["precision"], use_LPWSD
    )

imagen.max_embeddings_multiples = config.MAX_EMBEDDINGS_MULTIPLES

if scheduler := user_settings["scheduler"]:
    load_scheduler(scheduler)

if user_settings["safety_checker"] == "True":
    imagen.disable_safety_checker()

if (user_settings["attention_slicing"] == "True") or (
    user_settings["attention_slicing"] is None and imagen.device == "mps"
):  # Attention Slicing boosts performance on apple silicon
    # produces black image when fp16 and sdxl
    imagen.enable_attention_slicing()

for op in [
    "vae_slicing",
    "model_cpu_offload",
    "xformers_memory_attention",
]:
    if user_settings[op] == "True":
        getattr(imagen, f"enable_{op}")()

# Load embedding models.
# if sdxl and imagen.device == "mps":
#     # NotImplementedError: The operator 'aten::_linalg_eigvals' is not currently implemented for the MPS device.
#     logger.warn(
#         "Embedding models do not work with SDXL on the latest version of Diffusers. They will not be loaded."
#     )
# else:
for path in config.EMBEDDING_MODELS:
    logger.info(f"Loading embedding model {path}...")

    try:
        imagen.load_embedding_model(path)
    except OSError:
        logger.error(f"Embedding model {path} does not exist, skipping.")

if user_settings["compel_weighting"] == "True":
    if imagen.lpw_stable_diffusion_used:
        logger.warn(
            "Compel prompt weighting cannot be used when using LPWSD pipeline. Will not be enabled."
        )
    else:
        imagen.enable_compel_weighting()

# Load Loras.
for path in config.LORAS:
    logger.info(f"Loading lora {path}...")

    try:
        imagen.load_lora(path)
    except OSError as e:
        logger.error(f"Lora {path} does not exist, skipping.")


# Load theme.
theme = user_settings["theme"]

if len(theme) > 0:
    try:
        theme_manager.load_theme(theme)
    except ValueError:
        logger.error(f'Could not find theme "{theme}".')


texture_manager = TextureManager(dpg.add_texture_registry())
file_number = utils.next_file_number(config.SAVE_FILE_PATTERN)

# Widgets to disable when generating.
disable_widgets = None

base_image_aspect_ratio = None
last_step_latents = []
saves_tags = {}

# 0 is for generating
# 1 is interrupt called
# 2 is generation halted
# 3 is exit generation
gen_status = 0


def update_window_title(info: str = None):
    """Updates the window title with the specified information."""
    dpg.set_viewport_title(f"onGAU - {info}" if info else "onGAU")


def status(msg: str, log_func=logger.info):
    """Edits the status text and logs the message using the specified logging function."""
    if log_func:
        log_func(msg)

    dpg.set_value("status_text", msg)
    dpg.show_item("status_text")


def save_image_callback():
    """Callback to save the currently shown image to disk."""
    global file_number

    # Check for already existing files to not overwrite files.
    file_number = utils.next_file_number(config.SAVE_FILE_PATTERN, file_number)
    file_path = config.SAVE_FILE_PATTERN % file_number

    dpg.set_item_label("save_button", "Saving...")
    update_window_title(f"Saving to {file_path}...")

    utils.save_image(texture_manager.current()[1], file_path)

    dpg.set_item_label("save_button", "Save Image")
    update_window_title()

    # Add one to account for used file.
    file_number += 1


def save_model_callback():
    """Callback to save model weights to disk."""
    dpg.set_item_label("save_model", "Saving model..")
    update_window_title("Saving model...")
    logger.info("Saving model...")

    # Generate the path for model weights.
    dir_path = os.path.join(
        "models",
        ".".join(
            os.path.basename(imagen.model_path).split(".")[:-1]
        ),  # Get name of model.
    )
    os.mkdir(dir_path)
    imagen.save_weights(dir_path)

    logger.success(f"Saved model at {dir_path}.")
    dpg.set_item_label("save_model", "Save model weights")
    update_window_title()


def update_image_widget(texture_tag: str | int, image: GeneratedImage):
    """Updates output image widget with the specified texture."""
    # Resizes the image size to fit within window size.
    img_w, img_h = utils.resize_size_to_fit(
        (image.width, image.height),
        (
            # subtraction to account for position change
            dpg.get_viewport_width() - 440 - 7,
            dpg.get_viewport_height() - 60,  # subtraction to account for margin
        ),
    )

    if dpg.does_item_exist("output_image_item"):
        dpg.configure_item(
            "output_image_item",
            texture_tag=texture_manager.current()[0],
            width=img_w,
            height=img_h,
        )
    else:
        dpg.add_image(
            texture_tag,
            tag="output_image_item",
            before="output_image_group2",
            parent="output_image_group",
            width=img_w,
            height=img_h,
        )


def load_model(model_path: str):
    """Loads a new model."""
    status(f"Loading {model_path}...")
    update_window_title(f"Loading {model_path}...")

    try:
        imagen.set_model(
            model_path,
            use_lpw_stable_diffusion=imagen.lpw_stable_diffusion_used,
        )
    except (
        RuntimeError,
        FileNotFoundError,
    ) as e:  # When compel prompt weighting is enabled / model is not found.
        status(str(e), logger.error)
        update_window_title()
        return 1

    dpg.hide_item("status_text")
    update_window_title()
    return 0


def gen_progress_callback(
    step: int, step_count: int, step_time: float, elapsed_time, latents
):
    """Callback to update UI to show generation progress."""
    global last_step_latents, gen_status

    # step progress video mode
    # imagen.convert_latent_to_image(latents)[0].image.save(f"saves/{step}.png")

    if step == 1:
        last_step_latents.append(latents)

    # Check if generation has been interrupted.
    if gen_status == 1:
        # Status to generation halted.
        gen_status = 2

        # Continuously check for restart.
        while gen_status == 2:
            # Sleep to avoid using too much resources.
            time.sleep(0.1)

        # If exit generation is called.
        if gen_status == 3:
            raise pipelines.GenerationExit("Generation exited.", texture_manager.images)

        gen_status = 0

    last_step_latents[-1] = latents

    # Calculate the percentage
    progress = step / step_count
    eta = (step_count - step) * (elapsed_time / step)
    overlay = (
        f"{round(progress * 100)}% {step_time:.1f}s {step}/{step_count} ETA: {eta:.1f}s"
    )

    print(f"{GENERATING_MESSAGE}{logger.create(overlay, [logger.INFO])}")

    update_window_title(f"Generating... {overlay}")

    dpg.set_value("progress_bar", progress)
    dpg.configure_item(
        "progress_bar", overlay=overlay if progress < 1 else "Loading..."
    )


def prepare_UI():
    global disable_widgets

    dpg.show_item("gen_status_group")
    dpg.show_item("progress_bar")
    dpg.hide_item("info_group")
    dpg.hide_item("output_button_group")
    dpg.hide_item("output_image_group")
    dpg.hide_item("status_text")

    if disable_widgets is None:
        disable_widgets = [
            child
            for child in dpg.get_item_children("advanced_config")[1]
            if dpg.get_item_type(child) != "mvAppItemType::mvTooltip"
        ]
        disable_widgets.append("generate_btn")

    for child in disable_widgets:
        dpg.disable_item(child)


def finish_generation_callback(images: list, total_time: float, killed: bool = False):
    """Callback to run after generation thread finishes generation."""
    global last_step_latents

    last_step_latents = []

    for child in disable_widgets:
        dpg.enable_item(child)

    dpg.hide_item("gen_status_group")
    dpg.hide_item("progress_bar")

    # Reset progress bar..
    dpg.set_value("progress_bar", 0.0)
    dpg.configure_item("progress_bar", overlay="0%")

    if not images:
        return

    if isinstance(images, Exception):
        status(f"An error occurred during generation: {images}", None)
        logger.error(str(images))
        return

    # Add an "s" if there are more than 1 image.
    plural = "s" if len(images) > 1 else ""

    logger.success(f"Finished generating image{plural}.")

    average_step_time = total_time / (images[0].step_count * len(images))
    info = f"""Average step time: {average_step_time:.1f}s
Total time: {total_time:.1f}s"""

    logger.info(
        f"Seed{plural}: {', '.join([str(image.seed) for image in images])}\n{info}"
    )

    dpg.set_value("info_text", f"Current Image Seed: {images[0].seed}\n{info}")

    if not killed:
        # Prepare the images to be shown in UI.
        texture_manager.prepare(images)
        update_image_widget(*texture_manager.current())
    else:
        logger.success("Generation killed.")

    update_window_title()

    # Show image index counter.
    dpg.set_value("output_image_index", texture_manager.to_counter_string())

    dpg.show_item("info_group")
    dpg.show_item("output_button_group")
    dpg.show_item("output_image_group")


def generate_image_callback():
    """Callback to generate a new image."""
    texture_manager.clear()  # Save memory by deleting textures.

    # Get the path of the model.
    model_path = dpg.get_value("model")
    if model_path != imagen.model_path and load_model(model_path):
        return

    clip_skip = dpg.get_value("clip_skip")
    if clip_skip != imagen.clip_skip_amount:
        try:
            imagen.set_clip_skip_amount(clip_skip)
        except ValueError as e:
            logger.error(f"An error occurred while trying to set clip skip: {e}")

    # Start thread to generate image.
    if type(imagen) == Text2Img:
        pipelines.text2img(
            imagen, prepare_UI, finish_generation_callback, gen_progress_callback
        )
    else:
        pipelines.img2img(
            imagen, prepare_UI, finish_generation_callback, gen_progress_callback
        )


def switch_image_callback(tag: str):
    """Callback to switch through generated output images."""
    global image_index

    current = texture_manager.next() if tag == "next" else texture_manager.previous()

    if current:
        update_image_widget(*current)
        dpg.set_value("output_image_index", texture_manager.to_counter_string())

        try:
            seed = current[1].seed
        except AttributeError:
            seed = current[1].original_image.seed

        dpg.set_value(
            "info_text",
            f"Current Image Seed: {seed}\n{chr(10).join(dpg.get_value('info_text').splitlines()[1:])}",
        )


def checkbox_callback(tag: str, value: bool):
    """
    Callback for most checkbox settings.
    Enables and disables settings based on the tag.
    """
    if getattr(imagen, f"{tag}_enabled", value) == value:
        return

    func_name = f'{"enable_" if value else "disable_"}{tag}'

    try:
        getattr(imagen, func_name)()
    except Exception as e:
        status(str(e), logger.error)
        dpg.set_value(tag, not value)


def toggle_xformers_callback(_, value: bool):
    """Callback to toggle xformers."""
    if not torch.cuda.is_available():
        if value:
            dpg.set_value("xformers_memory_attention", False)
            status(
                "xformers memory attention is only available for CUDA GPUs.",
                logger.error,
            )

        return

    try:
        checkbox_callback("xformers_memory_attention", value)
    except ModuleNotFoundError:
        imagen.disable_xformers_memory_attention()
        dpg.set_value("xformers_memory_attention", False)
        status(
            "You don't have xformers installed. Please run `pip install xformers`.",
            logger.error,
        )


def change_pipeline_callback(_, pipeline: str):
    """Callback to change the pipeline used."""
    global imagen

    status(f"Loading {pipeline}...")
    update_window_title(f"Loading {pipeline}...")

    # Clear old imagen object.
    del imagen._pipeline

    if pipeline.startswith("SDXL"):
        imagen.sdxl = True

    match pipeline:
        case "Text2Img" | "SDXL Text2Img":
            imagen = Text2Img.from_class(imagen)
            dpg.hide_item("base_image_group")
            dpg.hide_item("strength_group")

        case "Img2Img" | "SDXL Img2Img":
            imagen = Img2Img.from_class(imagen)
            dpg.show_item("base_image_group")
            dpg.show_item("strength_group")
            base_image_path_callback()

    print(type(imagen._pipeline))

    dpg.hide_item("status_text")
    update_window_title()


def image_size_calc_callback(tag: str, value: str):
    """
    Callback to change the generated image width and height based on the aspect ratio of the base image.
    Only applies in img2img.
    """
    if base_image_aspect_ratio:
        if tag == "width":
            dpg.set_value("height", value / base_image_aspect_ratio)
        else:
            dpg.set_value("width", value * base_image_aspect_ratio)


def base_image_path_callback():
    """
    Callback to check if base image exists and assign generated image width and height to the size of the base image.
    Only applies in img2img.
    """
    global base_image_aspect_ratio

    base_image_path = dpg.get_value("base_image_path")
    if not os.path.isfile(base_image_path):
        status("Base image path does not exist.", None)
        return

    image_size = get_imsize(base_image_path)
    if image_size == (-1, -1):
        status("Base image path could not be read as a image file.", None)
        return

    base_image_aspect_ratio = image_size[0] / image_size[1]

    dpg.set_value("width", image_size[0])
    dpg.set_value("height", image_size[1])

    dpg.hide_item("status_text")  # remove any errors shown before


def lpwsd_callback(_, value: bool):
    """Callback to toggle Long Prompt Weighting Stable Diffusion pipeline."""
    status(f"Loading{' LPW' if value else ''} Stable Diffusion pipeline...")

    imagen.set_model(imagen.model_path, use_lpw_stable_diffusion=value)

    dpg.hide_item("status_text")  # remove any errors shown before


def use_in_img2img_callback():
    """Callback to send current generated image to img2img pipeline."""
    global file_number

    dpg.set_value("use_in_img2img_btn", "Loading...")

    file_number = utils.next_file_number(config.SAVE_FILE_PATTERN, file_number)
    file_path = config.SAVE_FILE_PATTERN % file_number

    utils.save_image(texture_manager.current()[1], file_path)

    # Add one to account for used file.
    file_number += 1

    dpg.set_value("base_image_path", file_path)

    # Change pipeline if not in img2img.
    if type(imagen) != Img2Img:
        pipeline = "SDXL Img2Img" if imagen.sdxl else "Img2Img"
        dpg.set_value("pipeline", pipeline)
        change_pipeline_callback(None, pipeline)

    dpg.set_value("use_in_img2img_btn", "Use In Img2Img")


def interrupt_callback():
    """Callback to interrupt the generation process."""
    global gen_status

    if gen_status == 2:
        gen_status = 0
        texture_manager.clear()

        logger.info("Generation restarted.")

        dpg.set_item_label("interrupt_btn", "Interrupt Generation")

        update_window_title()
        dpg.hide_item("output_button_group")

        dpg.hide_item("output_image_group")
        dpg.show_item("output_image_btns")
    else:
        gen_status = 1

        logger.info("Waiting for generation to stop...")
        update_window_title("Waiting for generation to stop...")

        # Wait for generation to actually stop to avoid segfaults.
        while gen_status == 1:
            time.sleep(0.2)

        logger.info("Decoding latents...")
        update_window_title("Decoding latents...")

        # Convert GeneratedLatents object into a GeneratedImage object to be compatible with other code.
        images = [
            imagen.convert_latent_to_image(latent)[0] for latent in last_step_latents
        ]

        dpg.set_item_label("interrupt_btn", "Continue Generation")

        texture_manager.prepare(images)
        update_image_widget(*texture_manager.current())
        update_window_title("Generation interrupted.")

        # Show image index counter.
        dpg.set_value("output_image_index", texture_manager.to_counter_string())

        dpg.show_item("output_button_group")
        dpg.show_item("output_image_group")


def load_settings(settings: dict):
    """Load settings from a dictionary."""
    for setting, value in settings.items():
        if not dpg.does_item_exist(setting):
            continue

        match setting:
            case "model":
                if value == imagen.model_path or load_model(value):
                    continue

                dpg.set_value("model", value)

            case "scheduler":
                load_scheduler(value)
                dpg.set_value(setting, value)

            case "pipeline":
                use_LPWSD = (
                    settings["lpwsd_pipeline"] == "True"
                    if "lpwsd_pipeline" in settings
                    else value == "StableDiffusionLongPromptWeightingPipeline"
                )

                same_pipeline = value in [
                    type(imagen).__name__,
                    imagen.pipeline.__name__,
                ]

                if same_pipeline and use_LPWSD == imagen.lpw_stable_diffusion_used:
                    continue

                if use_LPWSD:
                    # Force LPWSD pipeline without loading.
                    imagen.lpw_stable_diffusion_used = True

                # Convert pipeline class into imagen class.
                if value in [
                    "StableDiffusionPipeline",
                    "StableDiffusionLongPromptWeightingPipeline",
                    "Text2Img",
                ]:
                    if same_pipeline and use_LPWSD:
                        lpwsd_callback(None, True)
                        dpg.set_value("lpwsd_pipeline", True)
                    else:
                        change_pipeline_callback(None, "Text2Img")
                        dpg.set_value(setting, "Text2Img")
                elif value in ["StableDiffusionImg2ImgPipeline", "Img2Img"]:
                    change_pipeline_callback(None, "Img2Img")
                    dpg.set_value(setting, "Img2Img")
                else:
                    logger.error("Pipeline could not be understood.")

            case "loras":
                for lora in value.split(";"):
                    path = re.sub("\\(?=[,;])", "", lora)
                    imagen.load_lora(path)

            case "embeddings":
                embeds = value.split(";")

                # Fix for ", " joining for embeddings.
                if len(embeds) == 1:
                    embeds = value.split(", ")

                for embed in value.split(";"):
                    path = re.sub("\\(?=[,;])", "", embed)
                    imagen.load_embedding_model(path)

            case "clip_skip":
                imagen.set_clip_skip_amount(clip_skip := int(value))
                dpg.set_value("clip_skip", clip_skip)

            # case "lpwsd_pipeline":
            #     use_LPWSD = value == "True"
            #     if use_LPWSD == imagen.lpw_stable_diffusion_used:
            #         continue

            #     lpwsd_callback(None, use_LPWSD)

            case _:
                widget_type = dpg.get_item_type(setting)

                # Check the type of a widget to determine what type to cast to.
                if "Int" in widget_type:
                    dpg.set_value(setting, int(value))
                elif "Float" in widget_type:
                    dpg.set_value(setting, float(value))
                elif "Checkbox" in widget_type:
                    enabled = value == "True"

                    dpg.get_item_callback(setting)(setting, enabled)
                    dpg.set_value(setting, enabled)
                else:
                    dpg.set_value(setting, value)


def load_from_image_callback():
    """Callback to load generation settings from an inputted onGAU generated image file."""
    image_path = dpg.get_value("image_path_input")
    try:
        image = Image.open(image_path)
    except UnidentifiedImageError:
        status("Image does not exist or could not be read.", logger.error)
        return

    settings = image.info
    if dpg.get_value("ignore_model_checkbox"):
        settings.pop("model", None)

    if dpg.get_value("ignore_pipeline_checkbox"):
        settings.pop("lpwsd_pipeline", None)
        settings.pop("pipeline", None)

    width, height = image.size
    if "upscale_amount" in settings:
        up_amount = int(settings["upscale_amount"])
        width /= up_amount
        height /= up_amount

    dpg.set_value("width", width)
    dpg.set_value("height", height)

    dpg.hide_item("image_load_dialog")
    dpg.hide_item("status_text")

    load_settings(settings)


def kill_gen_callback():
    """Callback to kill the generation process."""
    global gen_status

    # Interrupt generation first
    interrupt_callback()
    dpg.set_item_label("interrupt_btn", "Interrupt Generation")

    # Set generation status to kill
    gen_status = 3


def update_delete_save_input():
    """Update the input items of the delete save dialog."""
    settings = settings_manager.settings
    dpg.configure_item(
        "delete_save_input",
        items=settings,
        default_value=settings[0] if len(settings) > 0 else "",
    )


def save_settings_callback():
    """Callback to save current settings into new save."""
    global saves_tags

    name = dpg.get_value("save_name_input")
    if name in ["DEFAULT", "main", ""]:
        logger.error(f'Cannot name a save "{name}".')
        return

    settings_manager.save_settings(
        name, ignore_keys=[] if dpg.get_value("include_model_checkbox") else ["model"]
    )

    update_delete_save_input()

    if name not in saves_tags:
        saves_tags[name] = dpg.add_menu_item(
            label=name,
            before="delete_save_button",
            callback=lambda: load_save(name),
            parent="saves_menu",
        )

    dpg.hide_item("save_settings_dialog")


def load_save(name: str):
    """Callback to load a save."""
    status(f"Loading save {name}...")

    load_settings(settings_manager.get_settings(name, full=True))

    dpg.hide_item("status_text")
    logger.success(f"Successfully loaded save {name}.")


def delete_save(name: str):
    """Callback to delete a save."""
    global saves_tags

    name = dpg.get_value("delete_save_input")
    if not name:
        logger.error("No saves to delete.")
        dpg.hide_item("delete_save_dialog")
        return

    settings_manager.delete_settings(name)
    update_delete_save_input()

    dpg.delete_item(saves_tags[name])
    del saves_tags[name]

    dpg.hide_item("delete_save_dialog")
    logger.success(f"Successfully deleted {name}.")


def reuse_seed_callback():
    """Callback to reuse seed of currently shown image for generation."""
    image = texture_manager.current()[1]

    try:
        seed = image.seed
    except AttributeError:
        seed = image.original_image.seed

    dpg.set_value("seed", seed)

    # if value := dpg.get_value("seed"):
    #     dpg.set_value("seed", seed)
    # else:
    #     dpg.set_value("seed", f"{value}, {seed}")


def reload_theme_callback():
    """Callback to reload themes."""
    logger.info("Reloading themes...")

    theme_manager.load_themes()

    for child in dpg.get_item_children("theme_buttons")[1]:
        dpg.delete_item(child)

    for name in theme_manager.get_themes():
        dpg.add_menu_item(
            label=name,
            callback=(lambda n: lambda: theme_manager.load_theme(n))(name),
            parent="theme_buttons",
        )


def precision_callback(_, precision: str):
    """Callback to set model precision."""
    logger.info(f"Loading {precision}...")
    imagen.set_precision(precision)
    logger.success(f"Loaded {precision}!")


def open_save_folder():
    """Callback to open save folder in FM."""
    path = os.path.abspath(os.path.dirname(config.SAVE_FILE_PATTERN))
    utils.open_path(path)


def open_models_folder():
    """Callback to open models folder in FM."""
    path = os.path.join(os.path.dirname(__file__), "models")
    utils.open_path(path)
