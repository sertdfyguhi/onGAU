from imagen import ImageGeneration, GeneratedImage
from PIL.PngImagePlugin import PngInfo
import dearpygui.dearpygui as dpg
import pyperclip
import config
import utils
import time
import os

# Constants
FILE_DIR = os.path.dirname(__file__)
MODEL = utils.append_dir_if_startswith(config.DEFAULT_MODEL, FILE_DIR, 'models/')
FONT = os.path.join(FILE_DIR, "fonts", config.FONT)


dpg.create_context()
dpg.create_viewport(title=config.WINDOW_TITLE, width=config.WINDOW_SIZE[0], height=config.WINDOW_SIZE[1])

imagen = ImageGeneration(MODEL, config.DEVICE)
imagen.disable_safety_checker()
imagen.enable_attention_slicing()

file_number = utils.next_file_number(config.SAVE_FILE_PATTERN)
last_step_time = None
current_seed = None


def save_image(image_info: GeneratedImage):
    global file_number

    metadata = PngInfo()
    metadata.add_text("model", image_info.model)
    metadata.add_text("prompt", image_info.prompt)
    metadata.add_text("negative_prompt", image_info.negative_prompt)
    metadata.add_text("strength", str(image_info.strength))
    metadata.add_text("guidance_scale", str(image_info.guidance_scale))
    metadata.add_text("step_count", str(image_info.step_count))
    metadata.add_text("seed", str(image_info.seed))

    image_info.image.save(config.SAVE_FILE_PATTERN % file_number, pnginfo=metadata)
    file_number += 1


def update_image(contents, width: int, height: int):
    with dpg.texture_registry():
        if dpg.does_item_exist("output_image"):
            dpg.delete_item("output_image")
            dpg.delete_item("output_image_item")

        dpg.add_static_texture(
            width=width,
            height=height,
            default_value=contents,
            tag="output_image",
        )

    # print("before image_widget_size calculation")

    image_widget_size = utils.resize_size_to_fit(
        (width, height),
        (
            dpg.get_viewport_width(),
            dpg.get_viewport_height() - 15,
        ),  # subtraction to account for margin
    )

    # print(image_widget_size)

    dpg.add_image(
        "output_image",
        pos=(460, 7),
        width=image_widget_size[0],
        height=image_widget_size[1],
        tag="output_image_item",
        parent="window",
    )


def progress_callback(step, step_count):
    global last_step_time

    elapsed_time = time.time() - last_step_time
    progress = step/ step_count
    overlay = f"{round(progress * 100)}% {elapsed_time:.1f}s {step}/{step_count}"

    print("generating...", overlay)

    dpg.set_viewport_title(f"{config.WINDOW_TITLE} - Generating {overlay}")

    dpg.set_value("progress_bar", progress)
    dpg.configure_item(
        "progress_bar", overlay=overlay if progress < 1 else "Loading..."
    )

    last_step_time = time.time()


def generate_image_callback():
    global current_seed, last_step_time

    model = utils.append_dir_if_startswith(dpg.get_value("model"), FILE_DIR, 'models/')
    if model != imagen.model:
        imagen.set_model(model)

    prompt = dpg.get_value("prompt")
    negative_prompt = dpg.get_value("negative_prompt")
    size = dpg.get_values(["image_width", "image_height"])
    strength = dpg.get_value("strength")
    guidance_scale = dpg.get_value("guidance_scale")
    step_count = dpg.get_value("step_count")
    seed = dpg.get_value("seed")
    if len(seed) == 0:
        seed = None
    else:
        seed = int(seed)

    if dpg.does_item_exist("output_image"):
        dpg.hide_item("output_image_item")
        dpg.set_value("output_image", [])

    dpg.hide_item("save_button")
    dpg.hide_item("seed_widget_group")
    dpg.show_item("progress_bar")

    last_step_time = start_time = time.time()

    print("generating image...")

    image = imagen.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        size=size,
        # strength / 100,
        guidance_scale=guidance_scale,
        step_count=step_count,
        seed=seed,
        progress_callback=lambda step, *_: progress_callback(step, step_count),
    )

    print("finished generating image; seed:", image.seed)

    dpg.set_viewport_title(config.WINDOW_TITLE)
    update_image(image.contents, image.width, image.height)

    current_seed = image.seed

    dpg.configure_item(
        "save_button",
        callback=lambda: save_image(image),
    )

    dpg.set_value("progress_bar", 0.0)
    dpg.configure_item("progress_bar", overlay="0%")
    dpg.set_value(
        "info_text", f"Seed: {image.seed}\nTotal time: {time.time() - start_time:.1f}s"
    )

    dpg.hide_item("progress_bar")
    dpg.show_item("seed_widget_group")
    dpg.show_item("save_button")


def safety_checker_callback(tag, value):
    if value:
        imagen.disable_safety_checker()
    else:
        imagen.enable_safety_checker()

def attention_slicing_callback(tag, value):
    if value:
        imagen.enable_attention_slicing()
    else:
        imagen.disable_attention_slicing()


# register font
with dpg.font_registry():
    default_font = dpg.add_font(FONT, config.FONT_SIZE)

with dpg.window(tag="window"):
    dpg.add_input_text(
        label="Model", default_value=config.DEFAULT_MODEL, width=config.ITEM_WIDTH, tag="model"
    )
    dpg.add_input_text(label="Prompt", width=config.ITEM_WIDTH, tag="prompt")
    dpg.add_input_text(label="Negative Prompt", width=config.ITEM_WIDTH, tag="negative_prompt")
    dpg.add_input_int(
        label="Width",
        default_value=512,
        min_value=1,
        width=config.ITEM_WIDTH,
        tag="image_width",
    )
    dpg.add_input_int(
        label="Height",
        default_value=512,
        min_value=1,
        width=config.ITEM_WIDTH,
        tag="image_height",
    )
    # dpg.add_input_int(
    #     label="Strength",
    #     default_value=80,
    #     max_value=100,
    #     width=config.ITEM_WIDTH,
    #     tag="strength",
    # )
    dpg.add_input_float(
        label="Guidance Scale",
        default_value=8.0,
        max_value=20.0,
        format="%.1f",
        width=config.ITEM_WIDTH,
        tag="guidance_scale",
    )
    dpg.add_input_int(
        label="Step Count",
        default_value=10,
        min_value=1,
        max_value=200,
        width=config.ITEM_WIDTH,
        tag="step_count",
    )
    dpg.add_input_text(
        label="Seed",
        width=config.ITEM_WIDTH,
        tag="seed",
    )
    dpg.add_checkbox(
        label="Disable Safety Checker",
        tag="safety_checker",
        default_value=True,
        callback=safety_checker_callback,
    )
    dpg.add_checkbox(
        label="Enable Attention Slicing",
        tag="attention_slicing",
        default_value=True,
        callback=attention_slicing_callback,
    )

    dpg.add_button(label="Generate Image", callback=generate_image_callback)
    dpg.add_progress_bar(overlay="0%", tag="progress_bar", width=config.ITEM_WIDTH, show=False)

    dpg.add_button(label="Save", tag="save_button", show=False)

    with dpg.group(horizontal=True, tag="seed_widget_group", show=False):
        dpg.add_text(tag="info_text")
        dpg.add_button(
            label="Copy Seed",
            callback=lambda: pyperclip.copy(current_seed),
        )

    dpg.bind_font(default_font)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("window", True)
print("starting GUI")
dpg.start_dearpygui()
dpg.destroy_context()
