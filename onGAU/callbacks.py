from imagen import Text2Img, SDImg2Img, ESRGAN, GeneratedImage, ESRGANUpscaledImage
from settings_manager import SettingsManager
from texture_manager import TextureManager
import logger, config, pipelines

from PIL import Image, UnidentifiedImageError
from diffusers import schedulers
import dearpygui.dearpygui as dpg
import imagesize
import utils
import torch
import time
import os
import re

dpg.create_context()

# Constants
FILE_DIR = os.path.dirname(__file__)  # get the directory path of this file
FONT = os.path.join(FILE_DIR, "fonts", config.FONT)

print(
    logger.create(
        f"Using device {logger.create(config.DEVICE, [logger.BOLD])}.", [logger.INFO]
    )
)


def load_scheduler(scheduler: str):
    """Load a scheduler."""
    scheduler_name = scheduler

    try:
        scheduler_name, use_karras = scheduler[: scheduler.index(" Karras")], True
    except ValueError:
        scheduler_name, use_karras = scheduler, False

    algorithm_type = (
        "dpmsolver"
        if scheduler_name == "DPMSolverMultistepScheduler"
        and not scheduler.endswith("++")
        else None
    )

    if (
        scheduler_name == imagen.scheduler.__class__.__name__
        and use_karras == imagen.karras_sigmas_used
        and algorithm_type == imagen.scheduler_algorithm_type
    ):
        return

    logger.info(f"Loading scheduler {scheduler}...")

    imagen.set_scheduler(
        getattr(schedulers, scheduler_name),
        use_karras_sigmas=use_karras,
        algorithm_type=algorithm_type,
    )


# load user settings
settings_manager = SettingsManager(config.USER_SETTINGS_FILE)
user_settings = settings_manager.get_settings("main")

use_LPWSD = user_settings["lpwsd_pipeline"] == "True"
model_path = utils.append_dir_if_startswith(user_settings["model"], FILE_DIR, "models/")
imagen_class = Text2Img if user_settings["pipeline"] == "Text2Img" else SDImg2Img


logger.info(f"Loading {model_path}...")

try:
    imagen = imagen_class(model_path, config.DEVICE, use_LPWSD)
except ValueError as e:
    logger.warn(str(e))

    # logger.info(f"Loading {model_path}...")
    imagen = imagen_class(model_path, config.DEVICE, False)
except FileNotFoundError:
    logger.error(f"{model_path} does not exist, falling back to default model.")

    model_path = utils.append_dir_if_startswith(
        config.DEFAULT_MODEL, FILE_DIR, "models/"
    )

    logger.info(f"Loading {model_path}...")
    imagen = imagen_class(model_path, config.DEVICE, use_LPWSD)

if user_settings["safety_checker"] == "True":
    imagen.disable_safety_checker()

if scheduler := user_settings["scheduler"]:
    load_scheduler(scheduler)

if (user_settings["attention_slicing"] == "True") or (
    user_settings["attention_slicing"] is None and imagen.device == "mps"
):  # Attention Slicing boosts performance on apple silicon
    imagen.enable_attention_slicing()

if user_settings["compel_weighting"] == "True":
    if imagen.lpw_stable_diffusion_used:
        logger.warn(
            "Compel prompt weighting cannot be used when using LPWSD pipeline. Will not be enabled."
        )
    else:
        imagen.enable_compel_weighting()

for op in [
    "vae_slicing",
    "xformers_memory_attention",
]:
    if user_settings[op] == "True":
        getattr(imagen, f"enable_{op}")()

# Load embedding models.
for path in config.EMBEDDING_MODELS:
    emb_model_path = utils.append_dir_if_startswith(path, FILE_DIR, "models/")
    logger.info(f"Loading embedding model {emb_model_path}...")

    try:
        imagen.load_embedding_model(emb_model_path)
    except OSError:
        logger.error(f"Embedding model {emb_model_path} does not exist, skipping.")

# Load Loras.
for path, weight in config.LORAS:
    lora_path = utils.append_dir_if_startswith(path, FILE_DIR, "models/")
    logger.info(f"Loading lora {lora_path}...")

    try:
        imagen.load_lora(lora_path, weight)
    except OSError as e:
        logger.error(f"Lora {lora_path} does not exist, skipping.")

texture_manager = TextureManager(dpg.add_texture_registry())
file_number = utils.next_file_number(config.SAVE_FILE_PATTERN)

base_image_aspect_ratio = None
saves_tags = {}
last_step_latents = []
esrgan = None

# 0 is for generating
# 1 is interrupt called
# 2 is generation halted
# 3 is exit generation
gen_status = 0


def update_window_title(info: str = None):
    """Updates the window title with the specified information."""
    dpg.set_viewport_title(
        f"{config.WINDOW_TITLE} - {info}" if info else config.WINDOW_TITLE
    )


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
        FILE_DIR,
        "models",
        os.path.basename(imagen.model).split(".")[0],  # Get name of model.
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
            before="output_image_selection",
            parent="output_image_group",
            width=img_w,
            height=img_h,
        )


def gen_progress_callback(step: int, step_count: int, elapsed_time: float, latents):
    """Callback to update UI to show generation progress."""
    global last_step_latents, gen_status

    if step == 0:
        last_step_latents.append(latents)

    # Check if generation has been interrupted.
    if gen_status == 1:
        # Status to generation halted.
        gen_status = 2

        # Continuously check for restart.
        while gen_status == 2:
            # Sleep to avoid using too much resources.
            time.sleep(1)

        # If exit generation is called.
        if gen_status == 3:
            raise RuntimeError("Generation exited.", texture_manager.images)

        gen_status = 0

    last_step_latents[-1] = latents

    # Calculate the percentage
    progress = step / step_count
    overlay = f"{round(progress * 100)}% {elapsed_time:.1f}s {step}/{step_count}"

    print(
        f"{logger.create('Generating... ', [logger.INFO, logger.BOLD])}{logger.create(overlay, [logger.INFO])}"
    )

    update_window_title(f"Generating... {overlay}")

    dpg.set_value("progress_bar", progress)
    dpg.configure_item(
        "progress_bar", overlay=overlay if progress < 1 else "Loading..."
    )


def load_model(model_path: str):
    """Loads a new model."""
    status(f"Loading {model_path}...")
    update_window_title(f"Loading {model_path}...")

    try:
        imagen.set_model(model_path, imagen.lpw_stable_diffusion_used)
    except (
        RuntimeError,
        FileNotFoundError,
    ) as e:  # When compel prompt weighting is enabled / model is not found.
        status(str(e), logger.error)
        update_window_title()
        return False
    except ValueError as e:  # When LPWSD pipeline is enabled.
        logger.warn(str(e))
        dpg.set_value("lpwsd_pipeline", False)
        imagen.set_model(model_path, False)

    dpg.hide_item("status_text")
    update_window_title()


def generate_image_callback():
    """Callback to generate a new image."""
    texture_manager.clear()  # Save memory by deleting textures.

    # Get the path of the model.
    model_path = utils.append_dir_if_startswith(
        dpg.get_value("model"), FILE_DIR, "models/"
    )
    if model_path != imagen.model and load_model(model_path):
        return

    scheduler = dpg.get_value("scheduler")
    load_scheduler(scheduler)

    clip_skip = dpg.get_value("clip_skip")
    if clip_skip != imagen.clip_skip_amount:
        try:
            imagen.set_clip_skip_amount(clip_skip)
        except ValueError as e:
            logger.error(str(e))

    dpg.show_item("gen_status_group")
    dpg.show_item("progress_bar")
    dpg.hide_item("info_group")
    dpg.hide_item("output_button_group")
    dpg.hide_item("output_image_group")
    dpg.hide_item("status_text")

    for child in dpg.get_item_children("advanced_config")[1]:
        # Ignore tooltips.
        if dpg.get_item_type(child) == "mvAppItemType::mvTooltip":
            continue

        dpg.disable_item(child)

    dpg.disable_item("generate_btn")

    def finish_generation_callback(
        images: list, total_time: float, killed: bool = False
    ):
        """Callback to run after generation thread finishes generation."""
        global last_step_latents

        last_step_latents = []

        if not images:
            return

        for child in dpg.get_item_children("advanced_config")[1]:
            # Ignore tooltips.
            if dpg.get_item_type(child) == "mvAppItemType::mvTooltip":
                continue

            dpg.enable_item(child)

        dpg.enable_item("generate_btn")

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

        # Reset progress bar..
        dpg.set_value("progress_bar", 0.0)
        dpg.configure_item("progress_bar", overlay="0%")

        # Show image index counter.
        dpg.set_value("output_image_index", texture_manager.to_counter_string())

        dpg.hide_item("gen_status_group")
        dpg.hide_item("progress_bar")
        dpg.show_item("info_group")
        dpg.show_item("output_button_group")
        dpg.show_item("output_image_group")

    # Start thread to generate image.
    if type(imagen) == Text2Img:
        pipelines.text2img(imagen, finish_generation_callback, gen_progress_callback)
    else:
        pipelines.img2img(imagen, finish_generation_callback, gen_progress_callback)


def switch_image_callback(tag: str):
    """Callback to switch through generated output images."""
    global image_index

    current = texture_manager.next() if tag == "next" else texture_manager.previous()

    if current:
        update_image_widget(*current)
        dpg.set_value("output_image_index", texture_manager.to_counter_string())
        dpg.set_value(
            "info_text",
            f"Current Image Seed: {current[1].seed}\n{chr(10).join(dpg.get_value('info_text').splitlines()[1:])}",
        )


def checkbox_callback(tag: str, value: bool):
    """
    Callback for most checkbox settings.
    Enables and disables settings based on the tag.
    """
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
            status("Xformers is only available for cuda.", logger.error)

        return

    try:
        checkbox_callback("xformers_memory_attention", value)
    except ModuleNotFoundError:
        imagen.disable_xformers_memory_attention()
        dpg.set_value("xformers_memory_attention", False)
        status(
            "You don't have xformers installed. Please run `pip3 install xformers`.",
            logger.error,
        )


def toggle_advanced_config_callback():
    """Callback to toggle visibility of advanced configurations."""
    if dpg.is_item_shown("advanced_config"):
        dpg.hide_item("advanced_config")
    else:
        dpg.show_item("advanced_config")


def change_pipeline_callback(_, pipeline: str):
    """Callback to change the pipeline used."""
    global imagen

    status(f"Loading {pipeline}...")
    update_window_title(f"Loading {pipeline}...")

    # Clear old imagen object.
    del imagen._pipeline

    match pipeline:
        case "Text2Img":
            imagen = Text2Img.from_class(imagen)
            dpg.hide_item("base_image_group")
        case "SDImg2Img":
            imagen = SDImg2Img.from_class(imagen)
            dpg.show_item("base_image_group")
            base_image_path_callback()

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

    image_size = imagesize.get(base_image_path)
    if image_size == (-1, -1):
        status("Base image path is not an image file.", None)
        return

    base_image_aspect_ratio = image_size[0] / image_size[1]

    dpg.set_value("width", image_size[0])
    dpg.set_value("height", image_size[1])

    dpg.hide_item("status_text")  # remove any errors shown before


def lpwsd_callback(_, value: bool):
    """Callback to toggle Long Prompt Weighting Stable Diffusion pipeline."""
    status(f"Loading{' LPW' if value else ''} Stable Diffusion pipeline...")

    try:
        imagen.set_model(imagen.model, value)
    except (RuntimeError, ValueError) as e:
        status(str(e), logger.error)
        dpg.set_value("lpwsd_pipeline", False)
        return

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
    if type(imagen) != SDImg2Img:
        dpg.set_value("pipeline", "SDImg2Img")
        change_pipeline_callback(None, "SDImg2Img")

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
        dpg.show_item("use_in_img2img_btn")
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

        # set_value doesn't work for some reason.
        dpg.set_item_label("interrupt_btn", "Continue Generation")

        texture_manager.prepare(images)
        update_image_widget(*texture_manager.current())
        update_window_title("Generation interrupted.")

        # Show image index counter.
        dpg.set_value("output_image_index", texture_manager.to_counter_string())

        dpg.show_item("output_button_group")

        dpg.hide_item("use_in_img2img_btn")
        dpg.show_item("output_image_group")


def upscale_image_callback():
    """Callback to upscale the currently shown image."""
    global esrgan

    # Check if it has been defined yet.
    if not config.ESRGAN_MODEL:
        logger.error(
            "ESRGAN model path has not been defined in config.py yet. Link to download the model: https://github.com/xinntao/Real-ESRGAN"
        )
        return

    # Initialize the ESRGAN model if it hasn't been initialized yet.
    if not esrgan:
        logger.info("Initializing ESRGAN model...")
        model_path = utils.append_dir_if_startswith(
            config.ESRGAN_MODEL, FILE_DIR, "models/"
        )

        try:
            esrgan = ESRGAN(model_path, config.DEVICE)
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
        esrgan, texture_manager.current()[1], dpg.get_value("upscale_amount"), callback
    )


def load_settings(settings: dict):
    """Load settings from a dictionary."""
    for setting, value in settings.items():
        if not dpg.does_item_exist(setting):
            continue

        if setting == "model" and value != imagen.model:
            load_model(value)
            dpg.set_value("model", value)
        elif setting == "scheduler":
            load_scheduler(value)
            dpg.set_value(setting, value)
        elif setting == "pipeline":
            use_LPWSD = settings["lpwsd_pipeline"] == "True"

            if (
                value in [imagen.__class__.__name__, imagen.pipeline.__name__]
                and use_LPWSD == imagen.lpw_stable_diffusion_used
            ):
                continue

            if use_LPWSD:
                imagen._lpw_stable_diffusion_used = True

            # Convert pipeline class into imagen class.
            match value:
                case "StableDiffusionPipeline" | "Text2Img":
                    change_pipeline_callback(None, "Text2Img")
                    dpg.set_value(setting, "Text2Img")

                case "StableDiffusionLongPromptWeightingPipeline":
                    # Force LPWSD pipeline without loading.
                    imagen._lpw_stable_diffusion_used = True

                    change_pipeline_callback(None, "Text2Img")
                    dpg.set_value(setting, "Text2Img")
                    dpg.set_value("lpwsd_pipeline", True)

                case "StableDiffusionImg2ImgPipeline" | "SDImg2Img":
                    change_pipeline_callback(None, "SDImg2Img")
                    dpg.set_value(setting, "SDImg2Img")

                case _:
                    logger.error("Pipeline could not be understood.")
        elif setting == "loras":
            # Split reformatted lora string.
            loras = [
                (
                    re.sub("\\(?=[,;])", "", lora.split(",")[0]),
                    float(lora.split(",")[1]),
                )
                for lora in value.split(";")
            ]

            for lora in loras:
                imagen.load_lora(*lora)
        elif setting == "embeddings":
            # Split reformatted embedding string.
            embeddings = [re.sub("\\(?=[,;])", "", value) for value in value.split(";")]

            for embedding in embeddings:
                imagen.load_embedding_model(embedding)
        elif setting == "clip_skip":
            imagen.set_clip_skip_amount(clip_skip := int(value))
            dpg.set_value("clip_skip", clip_skip)
        elif setting == "lpwsd_pipeline":
            use_LPWSD = value == "True"
            if use_LPWSD == imagen.lpw_stable_diffusion_used:
                continue

            lpwsd_callback(None, use_LPWSD)
        else:
            widget_type = dpg.get_item_type(setting)
            # print(setting, value)

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

    dpg.set_value("width", image.size[0])
    dpg.set_value("height", image.size[1])

    dpg.hide_item("image_load_dialog")
    dpg.hide_item("status_text")

    load_settings(image.info)


def kill_gen_callback():
    """Callback to kill the generation process."""
    global gen_status

    # Interrupt generation first
    interrupt_callback()
    dpg.set_item_label("interrupt_btn", "Interrupt Generation")

    # Set generation status to kill
    gen_status = 3


def load_save_callback(name: str):
    """Callback to load a save."""
    status(f"Loading save {name}...")

    load_settings(settings_manager.get_settings(name, full=True))

    dpg.hide_item("status_text")


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
    saves_tags[name] = dpg.add_menu_item(
        label=name,
        before="delete_save_button",
        callback=lambda: load_save_callback(name),
        parent="saves_menu",
    )
    dpg.hide_item("save_settings_dialog")


def delete_save_callback(name: str):
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