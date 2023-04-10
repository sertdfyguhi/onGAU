from imagen.text2img import Text2Img, GeneratedImage
from texture_manager import TextureManager
from PIL.PngImagePlugin import PngInfo
import dearpygui.dearpygui as dpg
from diffusers import schedulers
import pyperclip
import config
import torch
import utils
import time
import os
import re

# Constants
SCHEDULERS = [
    "DDIMInverseScheduler",
    "DDIMScheduler",
    "DDPMScheduler",
    "DEISMultistepScheduler",
    "DPMSolverMultistepScheduler",
    "DPMSolverSinglestepScheduler",
    "EulerAncestralDiscreteScheduler",
    "EulerDiscreteScheduler",
    "FlaxDDIMScheduler",
    "FlaxDDPMScheduler",
    "FlaxDPMSolverMultistepScheduler",
    "FlaxKarrasVeScheduler",
    "FlaxLMSDiscreteScheduler",
    "FlaxPNDMScheduler",
    "FlaxScoreSdeVeScheduler",
    "HeunDiscreteScheduler",
    "IPNDMScheduler",
    "KDPM2AncestralDiscreteScheduler",
    "KDPM2DiscreteScheduler",
    "KarrasVeScheduler",
    "LMSDiscreteScheduler",
    "PNDMScheduler",
    "RePaintScheduler",
    "ScoreSdeVeScheduler",
    "ScoreSdeVpScheduler",
    "UnCLIPScheduler",
    "UniPCMultistepScheduler",
    "VQDiffusionScheduler",
]
FILE_DIR = os.path.dirname(__file__)
MODEL = utils.append_dir_if_startswith(config.DEFAULT_MODEL, FILE_DIR, "models/")
FONT = os.path.join(FILE_DIR, "fonts", config.FONT)


dpg.create_context()
dpg.create_viewport(
    title=config.WINDOW_TITLE, width=config.WINDOW_SIZE[0], height=config.WINDOW_SIZE[1]
)

texture_manager = TextureManager(dpg.add_texture_registry())

imagen = Text2Img(MODEL, config.DEVICE)
imagen.disable_safety_checker()

if imagen.device == "mps":
    imagen.enable_attention_slicing()  # attention slicing boosts performance on m1 computers

file_number = utils.next_file_number(config.SAVE_FILE_PATTERN)
last_step_time = None


def update_window_title(info: str = None):
    dpg.set_viewport_title(
        f"{config.WINDOW_TITLE} - {info}" if info else config.WINDOW_TITLE
    )


def save_image(image_info: GeneratedImage):
    global file_number

    saved_file_path = config.SAVE_FILE_PATTERN % file_number

    dpg.set_item_label("save_button", "Saving...")
    update_window_title(f"Saving to {saved_file_path}...")

    metadata = PngInfo()
    metadata.add_text("model", image_info.model)
    metadata.add_text("prompt", image_info.prompt)
    metadata.add_text("negative_prompt", image_info.negative_prompt)
    metadata.add_text("strength", str(image_info.strength))
    metadata.add_text("guidance_scale", str(image_info.guidance_scale))
    metadata.add_text("step_count", str(image_info.step_count))
    metadata.add_text("pipeline", image_info.pipeline.__name__)
    metadata.add_text("scheduler", image_info.scheduler.__name__)
    metadata.add_text("seed", str(image_info.seed))

    image_info.image.save(config.SAVE_FILE_PATTERN % file_number, pnginfo=metadata)
    file_number += 1

    dpg.set_item_label("save_button", "Save Image")
    update_window_title()


def update_image(texture_tag: str | int, image: GeneratedImage):
    image_widget_size = utils.resize_size_to_fit(
        (image.width, image.height),
        (
            dpg.get_viewport_width(),
            dpg.get_viewport_height() - 42,
        ),  # subtraction to account for margin
    )

    if dpg.does_item_exist("output_image_item"):
        dpg.configure_item(
            "output_image_item",
            texture_tag=texture_manager.current()[0],
            width=image_widget_size[0],
            height=image_widget_size[1],
        )
    else:
        dpg.add_image(
            texture_tag,
            width=image_widget_size[0],
            height=image_widget_size[1],
            tag="output_image_item",
            before="output_image_selection",
            parent="output_image_group",
        )

    dpg.configure_item(
        "save_button",
        callback=lambda: save_image(image),
    )


def progress_callback(step, step_count):
    global last_step_time

    elapsed_time = time.time() - last_step_time
    progress = step / step_count
    overlay = f"{round(progress * 100)}% {elapsed_time:.1f}s {step}/{step_count}"

    print("generating...", overlay)

    update_window_title(f"Generating... {overlay}")

    dpg.set_value("progress_bar", progress)
    dpg.configure_item(
        "progress_bar", overlay=overlay if progress < 1 else "Loading..."
    )

    last_step_time = time.time()


def generate_image_callback():
    global last_step_time

    model = utils.append_dir_if_startswith(dpg.get_value("model"), FILE_DIR, "models/")
    if model != imagen.model:
        dpg.show_item("info_text")
        dpg.set_value("info_text", f"Loading {model}...")
        update_window_title(f"Loading {model}...")

        imagen.set_model(model)

        dpg.hide_item("info_text")
        update_window_title()

    scheduler = dpg.get_value("scheduler")
    if scheduler != imagen.scheduler.__name__:
        imagen.set_scheduler(getattr(schedulers, scheduler))

    prompt = dpg.get_value("prompt")
    negative_prompt = dpg.get_value("negative_prompt")
    size = dpg.get_values(["image_width", "image_height"])
    # strength = dpg.get_value("strength")
    guidance_scale = dpg.get_value("guidance_scale")
    step_count = dpg.get_value("step_count")
    image_amount = dpg.get_value("image_amount")
    seed = dpg.get_value("seed")
    if seed:
        try:
            seed = (
                int(seed)
                if seed.isdigit()
                else [int(s) for s in re.split(r"[, ]+", seed)]
            )
        except ValueError:
            dpg.set_value("info_text", "seeds provided are not integers")
            dpg.show_item("info_text")
            return

    dpg.show_item("progress_bar")
    dpg.hide_item("save_button")
    dpg.hide_item("info_text")
    dpg.hide_item("seed_button")
    dpg.hide_item("output_image_group")

    last_step_time = start_time = time.time()

    print("generating image...")

    texture_manager.prepare(
        imagen.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            size=size,
            # strength=strength,
            guidance_scale=guidance_scale,
            step_count=step_count,
            seed=seed,
            image_amount=image_amount,
            progress_callback=lambda step, *_: progress_callback(step, step_count),
        )
    )

    print(
        "finished generating image; seeds:",
        ", ".join([str(image.seed) for image in texture_manager.images]),
    )

    update_window_title()
    update_image(*texture_manager.current())

    dpg.set_value("progress_bar", 0.0)
    dpg.configure_item("progress_bar", overlay="0%")
    dpg.set_value("output_image_index", texture_manager.to_counter_string())
    dpg.set_value(
        "info_text",
        f"Current Image Seed: {texture_manager.images[0].seed}\nTotal time: {time.time() - start_time:.1f}s",
    )

    dpg.hide_item("progress_bar")
    dpg.show_item("info_text")
    dpg.show_item("seed_button")
    dpg.show_item("save_button")
    dpg.show_item("output_image_group")


def change_image(tag):
    global image_index

    if tag == "next":
        current = texture_manager.next()
    else:
        current = texture_manager.previous()

    if current:
        update_image(*current)
        dpg.set_value("output_image_index", texture_manager.to_counter_string())
        dpg.set_value(
            "info_text",
            f"Current Image Seed: {current[1].seed}\n{chr(10).join(dpg.get_value('info_text').splitlines()[1:])}",
        )


def checkbox_callback(tag, value):
    func_name = ("enable_" if value else "disable_") + tag
    getattr(imagen, func_name)()


def toggle_xformers(tag, value):
    if not torch.cuda.is_available():
        dpg.set_value("xformers_memory_attention", False)
        dpg.show_item("info_text")
        dpg.set_value("info_text", "xformers is only available for GPUs")
        print("xformers is only available for GPUs")
        return

    try:
        checkbox_callback(tag, value)
    except ModuleNotFoundError:
        imagen.disable_xformers_memory_attention()
        dpg.set_value("xformers_memory_attention", False)
        dpg.show_item("info_text")
        dpg.set_value(
            "info_text", "xformers is not installed, please run `pip3 install xformers`"
        )
        print(
            "xformers is not installed, please run \033[1mpip3 install xformers\033[0m"
        )


def toggle_advanced_config():
    if dpg.is_item_shown("advanced_config"):
        dpg.hide_item("advanced_config")
    else:
        dpg.show_item("advanced_config")


# register font
with dpg.font_registry():
    default_font = dpg.add_font(FONT, config.FONT_SIZE)

with dpg.window(tag="window"):
    dpg.add_input_text(
        label="Model",
        default_value=config.DEFAULT_MODEL,
        width=config.ITEM_WIDTH,
        tag="model",
    )
    dpg.add_input_text(label="Prompt", width=config.ITEM_WIDTH, tag="prompt")
    dpg.add_input_text(
        label="Negative Prompt", width=config.ITEM_WIDTH, tag="negative_prompt"
    )
    dpg.add_input_int(
        label="Width",
        default_value=config.DEFAULT_IMAGE_SIZE[0],
        min_value=1,
        width=config.ITEM_WIDTH,
        tag="image_width",
    )
    dpg.add_input_int(
        label="Height",
        default_value=config.DEFAULT_IMAGE_SIZE[1],
        min_value=1,
        width=config.ITEM_WIDTH,
        tag="image_height",
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
        default_value=8.0,
        max_value=50.0,
        format="%.1f",
        width=config.ITEM_WIDTH,
        tag="guidance_scale",
    )
    dpg.add_input_int(
        label="Step Count",
        default_value=20,
        min_value=1,
        max_value=500,
        width=config.ITEM_WIDTH,
        tag="step_count",
    )
    dpg.add_input_int(
        label="Amount of Images",
        default_value=1,
        min_value=1,
        max_value=100,
        width=config.ITEM_WIDTH,
        tag="image_amount",
    )
    dpg.add_input_text(
        label="Seed",
        width=config.ITEM_WIDTH,
        tag="seed",
    )
    dpg.add_button(label="Advanced Configuration", callback=toggle_advanced_config)

    with dpg.group(tag="advanced_config", show=False):
        dpg.add_combo(
            label="Scheduler",
            items=SCHEDULERS,
            default_value=imagen.scheduler.__name__,
            width=config.ITEM_WIDTH,
            tag="scheduler",
        )
        dpg.add_checkbox(
            label="Disable Safety Checker",
            tag="safety_checker",
            default_value=not imagen.safety_checker_enabled,
            callback=lambda tag, value: checkbox_callback(tag, not value),
        )
        dpg.add_checkbox(
            label="Enable Attention Slicing",
            tag="attention_slicing",
            default_value=imagen.attention_slicing_enabled,
            callback=checkbox_callback,
        )
        dpg.add_checkbox(
            label="Enable Vae Slicing",
            tag="vae_slicing",
            default_value=imagen.vae_slicing_enabled,
            callback=checkbox_callback,
        )
        dpg.add_checkbox(
            label="Enable xFormers Memory Efficient Attention",
            tag="xformers_memory_attention",
            default_value=imagen.xformers_memory_attention_enabled,
            callback=toggle_xformers,
        )

    dpg.add_button(label="Generate Image", callback=generate_image_callback)
    dpg.add_progress_bar(
        overlay="0%", tag="progress_bar", width=config.ITEM_WIDTH, show=False
    )

    dpg.add_button(label="Save Image", tag="save_button", show=False)

    with dpg.group(horizontal=True):
        dpg.add_text(tag="info_text", show=False)
        dpg.add_button(
            label="Copy Seed",
            tag="seed_button",
            callback=lambda: pyperclip.copy(texture_manager.current()[1].seed),
            show=False,
        )

    with dpg.group(pos=(460, 7), show=False, tag="output_image_group"):
        with dpg.group(horizontal=True, tag="output_image_selection"):
            dpg.add_button(label="<", tag="previous", callback=change_image)
            dpg.add_button(label=">", tag="next", callback=change_image)
            dpg.add_text(tag="output_image_index")

    dpg.bind_font(default_font)

dpg.set_primary_window("window", True)

if __name__ == "__main__":
    print("starting GUI")
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
