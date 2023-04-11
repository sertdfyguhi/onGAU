from imagen.text2img import Text2Img, GeneratedImage
import dearpygui.dearpygui as dpg
from diffusers import schedulers
from typing import Callable
import time
import re

last_step_time = None


def _callback(step: int, step_count: int, progress_callback: Callable):
    """calculates the time of the last step and calls progress callback"""
    global last_step_time

    progress_callback(step, step_count, time.time() - last_step_time)
    last_step_time = time.time()


def text2img(imagen: Text2Img, progress_callback: Callable) -> list[GeneratedImage]:
    global last_step_time

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
            dpg.hide_item("progress_bar")
            return

    last_step_time = time.time()

    print("generating image...")

    return imagen.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        size=size,
        # strength=strength,
        guidance_scale=guidance_scale,
        step_count=step_count,
        seed=seed,
        image_amount=image_amount,
        progress_callback=lambda step, *_: _callback(
            step, step_count, progress_callback
        ),
    )
