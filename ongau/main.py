from imagen import Text2Img, SDImg2Img, GeneratedImage
from texture_manager import TextureManager
from user_settings import UserSettings
import logger, config, pipelines

from diffusers import schedulers
import dearpygui.dearpygui as dpg
import imagesize
import pyperclip
import utils
import torch
import time
import os

# Constants
FILE_DIR = os.path.dirname(__file__)  # get the directory path of this file
FONT = os.path.join(FILE_DIR, "fonts", config.FONT)
DEVICE = config.DEVICE

if DEVICE == "auto":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif getattr(torch, "has_mps", False):
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

print(
    logger.create(
        f"Using device {logger.create(DEVICE, [logger.BOLD])}.", [logger.INFO]
    )
)

# load user settings
settings_manager = UserSettings(config.USER_SETTINGS_FILE)
user_settings = settings_manager.get_user_settings()

use_LPWSD = user_settings["lpwsd_pipeline"] == "True"
model_path = utils.append_dir_if_startswith(user_settings["model"], FILE_DIR, "models/")
imagen_class = Text2Img if user_settings["pipeline"] == "Text2Img" else SDImg2Img


logger.info(f"Loading {model_path}...")

try:
    imagen = imagen_class(model_path, DEVICE, use_LPWSD)
except ValueError as e:
    logger.warn(str(e))

    logger.info(f"Loading {model_path}...")
    imagen = imagen_class(model_path, DEVICE, False)
except FileNotFoundError:
    logger.error(f"{model_path} does not exist, falling back to default model.")

    model_path = utils.append_dir_if_startswith(
        config.DEFAULT_MODEL, FILE_DIR, "models/"
    )

    logger.info(f"Loading {model_path}...")
    imagen = imagen_class(model_path, DEVICE, use_LPWSD)

if user_settings["safety_checker"] == "True":
    imagen.disable_safety_checker()

if scheduler := user_settings["scheduler"]:
    # Check if scheduler is using karras sigmas by checking if it endswith "Karras".
    if scheduler.endswith("Karras"):
        imagen.set_scheduler(getattr(schedulers, scheduler[:-7]), True)
    else:
        imagen.set_scheduler(getattr(schedulers, scheduler))

if user_settings["attention_slicing"] != None:
    if user_settings["attention_slicing"] == "True":
        imagen.enable_attention_slicing()
else:
    # Attention slicing boosts performance on Apple Silicon.
    if imagen.device == "mps":
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
        getattr(imagen, "enable_" + op)()

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

dpg.create_context()
dpg.create_viewport(
    title=config.WINDOW_TITLE, width=config.WINDOW_SIZE[0], height=config.WINDOW_SIZE[1]
)

texture_manager = TextureManager(dpg.add_texture_registry())
file_number = utils.next_file_number(config.SAVE_FILE_PATTERN)
base_image_aspect_ratio = None


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

    # Generate the path for model weights.
    dir_path = os.path.join(
        FILE_DIR,
        "models",
        os.path.basename(imagen.model).split(".")[0],  # Get name of model.
    )
    os.mkdir(dir_path)
    imagen.save_weights(dir_path)

    dpg.set_item_label("save_model", "Save model weights")
    update_window_title()


def update_image_widget(texture_tag: str | int, image: GeneratedImage):
    """Updates output image widget with the specified texture."""
    # Resizes the image size to fit within window size.
    img_w, img_h = utils.resize_size_to_fit(
        (image.width, image.height),
        (
            dpg.get_viewport_width() - 460,  # subtration to account for position change
            dpg.get_viewport_height() - 42,  # subtraction to account for margin
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


def gen_progress_callback(step: int, step_count: int, elapsed_time: float):
    """Callback to update UI to show generation progress."""
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


def generate_image_callback():
    """Callback to generate a new image."""
    texture_manager.clear()  # save memory

    # Get the path of the model.
    model_path = utils.append_dir_if_startswith(
        dpg.get_value("model"), FILE_DIR, "models/"
    )
    if model_path != imagen.model:
        status(f"Loading {model_path}...")
        update_window_title(f"Loading {model_path}...")

        try:
            imagen.set_model(model_path, imagen.lpw_stable_diffusion_used)
        except RuntimeError as e:  # When compel prompt weighting is enabled.
            status(str(e), logger.error)
            update_window_title()
            return
        except ValueError as e:  # When LPWSD pipeline is enabled.
            logger.warn(str(e))
            imagen.set_model(model_path, False)

        dpg.hide_item("status_text")
        update_window_title()

    scheduler = dpg.get_value("scheduler")

    # Check if scheduler is different from currently used scheduler.
    if scheduler != imagen.scheduler.__name__ + (
        " Karras" if imagen.karras_sigmas_used else ""
    ):
        logger.info(f"Loading scheduler {scheduler}...")

        use_karras = scheduler.endswith("Karras")
        imagen.set_scheduler(
            getattr(schedulers, scheduler[:-7] if use_karras else scheduler),
            use_karras,
        )

    clip_skip = dpg.get_value("clip_skip")
    if clip_skip != imagen.clip_skip_amount:
        try:
            imagen.set_clip_skip_amount(clip_skip)
        except ValueError as e:
            logger.error(str(e))

    dpg.show_item("progress_bar")
    dpg.hide_item("info_group")
    dpg.hide_item("output_button_group")
    dpg.hide_item("output_image_group")
    dpg.hide_item("status_text")

    start_time = time.time()

    if type(imagen) == Text2Img:
        images = pipelines.text2img(imagen, gen_progress_callback)
    else:
        images = pipelines.img2img(imagen, gen_progress_callback)

    if not images:
        return

    # Add an "s" if there are more than 1 image.
    plural = "s" if len(images) > 1 else ""
    total_time = time.time() - start_time

    print(
        logger.create(
            f"Finished generating image{plural}.", [logger.SUCCESS, logger.BOLD]
        )
    )

    average_step_time = total_time / sum([image.step_count for image in images])

    logger.info(
        f"""Seed{plural}: {', '.join([str(image.seed) for image in images])}
Average step time: {average_step_time:.1f}s
Total time: {total_time:.1f}s"""
    )

    dpg.set_value(
        "info_text",
        f"Current Image Seed: {images[0].seed}\nAverage step time: {average_step_time:.1f}s\nTotal time: {total_time:.1f}s",
    )

    # Prepare the images to be shown in UI.
    texture_manager.prepare(images)

    update_window_title()
    update_image_widget(*texture_manager.current())

    # Reset progress bar..
    dpg.set_value("progress_bar", 0.0)
    dpg.configure_item("progress_bar", overlay="0%")

    # Show image index counter.
    dpg.set_value("output_image_index", texture_manager.to_counter_string())

    dpg.hide_item("progress_bar")
    dpg.show_item("info_group")
    dpg.show_item("output_button_group")
    dpg.show_item("output_image_group")


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
    func_name = ("enable_" if value else "disable_") + tag

    try:
        getattr(imagen, func_name)()
    except Exception as e:
        status(str(e), logger.error)
        dpg.set_value(tag, not value)


def toggle_xformers_callback(_, value: bool):
    """Callback to toggle xformers."""
    if not torch.cuda.is_available():
        dpg.set_value("xformers_memory_attention", False)
        status("Xformers is only available for GPUs.", logger.error)
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

    dpg.set_value("use_in_img2img", "Loading...")

    file_number = utils.next_file_number(config.SAVE_FILE_PATTERN, file_number)
    file_path = config.SAVE_FILE_PATTERN % file_number

    # Add one to account for used file.
    file_number += 1

    utils.save_image(texture_manager.current()[1], file_path)

    dpg.set_value("base_image_path", file_path)

    # Change pipeline if not in img2img.
    if type(imagen) != SDImg2Img:
        dpg.set_value("pipeline", "SDImg2Img")
        change_pipeline_callback(None, "SDImg2Img")

    dpg.set_value("use_in_img2img", "Use In Img2Img")


# def load_settings_image_callback():
#     """Callback to load generation settings from an inputted onGAU generated image file."""
#     pass


# Register UI font.
with dpg.font_registry():
    default_font = dpg.add_font(FONT, config.FONT_SIZE)

# Register key shortcuts.
with dpg.handler_registry():
    dpg.add_key_down_handler(
        dpg.mvKey_Left, callback=lambda: switch_image_callback("previous")
    )
    dpg.add_key_down_handler(
        dpg.mvKey_Right, callback=lambda: switch_image_callback("next")
    )

# Create dialog box for loading settings from an image file.
# with dpg.window(tag="image_input_dialog", no_title_bar=True, modal=True):
#     dpg.add_text("Image path.")
#     dpg.add_input_text(tag="image_input", width=config.ITEM_WIDTH)
#     with dpg.group(horizontal=True):
#         dpg.add_button(label="Cancel")
#         dpg.add_button(label="Ok")

with dpg.window(tag="window"):
    # with dpg.menu_bar():
    #     with dpg.menu(label="File"):
    #         dpg.add_menu_item(
    #             label="Load Settings from Image", callback=load_settings_image_callback
    #         )

    dpg.add_input_text(
        label="Model",
        default_value=model_path,
        width=config.ITEM_WIDTH,
        tag="model",
    )
    dpg.add_text(
        "The path to a Stable Diffusion model to use. (huggingface model or local model)",
        parent=dpg.add_tooltip("model"),
    )

    dpg.add_input_text(
        label="Prompt",
        default_value=user_settings["prompt"],
        width=config.ITEM_WIDTH,
        tag="prompt",
    )
    dpg.add_text(
        "The instructions of the generated image.",
        parent=dpg.add_tooltip("prompt"),
    )

    dpg.add_input_text(
        label="Negative Prompt",
        default_value=user_settings["negative_prompt"],
        width=config.ITEM_WIDTH,
        tag="negative_prompt",
    )
    dpg.add_text(
        "The instructions of what to remove of the generated image.",
        parent=dpg.add_tooltip("negative_prompt"),
    )

    dpg.add_input_int(
        label="Width",
        default_value=int(user_settings["width"]),
        min_value=1,
        width=config.ITEM_WIDTH,
        callback=image_size_calc_callback,
        tag="width",
    )
    dpg.add_text(
        "The image width of the output image.",
        parent=dpg.add_tooltip("width"),
    )

    dpg.add_input_int(
        label="Height",
        default_value=int(user_settings["height"]),
        min_value=1,
        width=config.ITEM_WIDTH,
        callback=image_size_calc_callback,
        tag="height",
    )
    dpg.add_text(
        "The image height of the output image.",
        parent=dpg.add_tooltip("height"),
    )

    # dpg.add_input_float(
    #     label="Strength",
    #     default_value=0.8,
    #     min_value=0.0,
    #     max_value=1.0,
    #     format="%.1f",
    #     width=config.ITEM_WIDTH,
    #     tag="strength",
    # )
    dpg.add_input_float(
        label="Guidance Scale",
        default_value=float(user_settings["guidance_scale"]),
        min_value=0.0,
        max_value=50.0,
        format="%.1f",
        width=config.ITEM_WIDTH,
        tag="guidance_scale",
    )
    dpg.add_text(
        "How closely SD should follow your prompt.",
        parent=dpg.add_tooltip("guidance_scale"),
    )

    dpg.add_input_int(
        label="Step Count",
        default_value=int(user_settings["step_count"]),
        min_value=1,
        max_value=500,
        width=config.ITEM_WIDTH,
        tag="step_count",
    )
    dpg.add_text(
        "The number of iterations that SD runs over the image. Higher number usually gets you a better image.",
        parent=dpg.add_tooltip("step_count"),
    )

    dpg.add_input_int(
        label="Amount of Images",
        default_value=int(user_settings["image_amount"]),
        min_value=1,
        max_value=100,
        width=config.ITEM_WIDTH,
        tag="image_amount",
    )
    dpg.add_text(
        "The amount of images to generate.",
        parent=dpg.add_tooltip("image_amount"),
    )

    dpg.add_input_text(
        label="Seed",
        default_value=user_settings["seed"],
        width=config.ITEM_WIDTH,
        tag="seed",
    )
    dpg.add_text(
        "The number to initialize the generation. Leaving it empty chooses it randomly.",
        parent=dpg.add_tooltip("seed"),
    )

    # file dialog to be used
    # with dpg.group(horizontal=True):
    #     btn = dpg.add_button(label="Choose...")
    # Group to hide and show tooltip as it breaks if you only hide the parent.
    with dpg.group(
        tag="base_image_group", show=user_settings["pipeline"] == "SDImg2Img"
    ):
        dpg.add_input_text(
            # readonly=True,
            label="Base Image Path",
            default_value=user_settings["base_image_path"],
            width=config.ITEM_WIDTH,
            tag="base_image_path",
            callback=base_image_path_callback,
        )
        dpg.add_text(
            "The path of the starting image to use in img2img.",
            parent=dpg.add_tooltip("base_image_path"),
        )

    dpg.add_button(
        label="Advanced Configuration", callback=toggle_advanced_config_callback
    )

    with dpg.group(tag="advanced_config", indent=7, show=False):
        dpg.add_combo(
            label="Pipeline",
            items=["Text2Img", "SDImg2Img"],
            default_value=imagen.__class__.__name__,
            width=config.ITEM_WIDTH,
            callback=change_pipeline_callback,
            tag="pipeline",
        )
        dpg.add_text(
            "The pipeline to use.",
            parent=dpg.add_tooltip("pipeline"),
        )

        dpg.add_combo(
            label="Scheduler",
            items=config.SCHEDULERS,
            default_value=user_settings["scheduler"]
            if user_settings["scheduler"]
            else imagen.scheduler.__name__,
            width=config.ITEM_WIDTH,
            tag="scheduler",
        )
        dpg.add_text(
            "The sampling method to use.",
            parent=dpg.add_tooltip("scheduler"),
        )

        dpg.add_input_int(
            label="Clip Skip",
            default_value=int(user_settings["clip_skip"]),
            width=config.ITEM_WIDTH,
            tag="clip_skip",
        )
        dpg.add_text(
            "The amount CLIP layers to skip.",
            parent=dpg.add_tooltip("clip_skip"),
        )

        dpg.add_checkbox(
            label="Disable Safety Checker",
            tag="safety_checker",
            default_value=not imagen.safety_checker_enabled,
            callback=lambda tag, value: checkbox_callback(tag, not value),
        )
        dpg.add_text(
            "Check for NSFW image.",
            parent=dpg.add_tooltip("safety_checker"),
        )

        dpg.add_checkbox(
            label="Enable Attention Slicing",
            tag="attention_slicing",
            default_value=imagen.attention_slicing_enabled,
            callback=checkbox_callback,
        )
        dpg.add_text(
            "Slices the computation into multiple steps. Increases performance on MPS.",
            parent=dpg.add_tooltip("attention_slicing"),
        )

        dpg.add_checkbox(
            label="Enable Vae Slicing",
            tag="vae_slicing",
            default_value=imagen.vae_slicing_enabled,
            callback=checkbox_callback,
        )
        dpg.add_text(
            "VAE decodes one image at a time.",
            parent=dpg.add_tooltip("vae_slicing"),
        )

        dpg.add_checkbox(
            label="Enable xFormers Memory Efficient Attention",
            tag="xformers_memory_attention",
            default_value=imagen.xformers_memory_attention_enabled,
            callback=toggle_xformers_callback,
        )

        dpg.add_checkbox(
            label="Enable Compel Prompt Weighting",
            tag="compel_weighting",
            default_value=imagen.compel_weighting_enabled,
            callback=checkbox_callback,
        )
        dpg.add_text(
            "Use compel prompt weighting. + to increase weight and - to decrease.",
            parent=dpg.add_tooltip("compel_weighting"),
        )

        dpg.add_checkbox(
            label="Enable LPW Stable Diffusion Pipeline",
            tag="lpwsd_pipeline",
            default_value=imagen.lpw_stable_diffusion_used,
            callback=lpwsd_callback,
        )
        dpg.add_text(
            "Use LPWSD pipeline. Adds prompt weighting as seen in A1111's webui and long prompts.",
            parent=dpg.add_tooltip("lpwsd_pipeline"),
        )

        dpg.add_button(
            label="Save model weights",
            tag="save_model",
            callback=save_model_callback,
        )

    dpg.add_button(label="Generate Image", callback=generate_image_callback)
    dpg.add_progress_bar(
        overlay="0%", tag="progress_bar", width=config.ITEM_WIDTH, show=False
    )

    # change tag name to smth better
    with dpg.group(tag="output_button_group", show=False):
        dpg.add_button(
            label="Save Image", tag="save_button", callback=save_image_callback
        )
        dpg.add_button(
            label="Use In Img2Img",
            tag="use_in_img2img",
            callback=use_in_img2img_callback,
        )

    with dpg.group(horizontal=True, tag="info_group", show=False):
        dpg.add_text(tag="info_text")
        dpg.add_button(
            label="Copy Seed",
            callback=lambda: pyperclip.copy(texture_manager.current()[1].seed),
        )

    dpg.add_text(tag="status_text")

    with dpg.group(pos=(460, 7), show=False, tag="output_image_group"):
        with dpg.group(horizontal=True, tag="output_image_selection"):
            dpg.add_button(label="<", tag="previous", callback=switch_image_callback)
            dpg.add_button(label=">", tag="next", callback=switch_image_callback)
            dpg.add_text(tag="output_image_index")

    dpg.bind_font(default_font)

dpg.set_primary_window("window", True)

if __name__ == "__main__":
    dpg.set_exit_callback(settings_manager.save_user_settings)

    logger.success("Starting GUI...")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
