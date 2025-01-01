from imagen import *
import logger

from PIL import UnidentifiedImageError, Image
from threading import Thread
import dearpygui.dearpygui as dpg
import traceback
import time
import re
import os


class GenerationExit(BaseException):
    pass


def thread_generate(func, callback, **kwargs):
    logger.info("Starting generation...")

    def worker():
        start_time = time.time()

        try:
            images = func(**kwargs)
        except GenerationExit as e:  #
            callback(e.args[1], time.time() - start_time, True)
            return
        except Exception as e:
            traceback.print_exc(e)
            callback(e, time.time() - start_time, True)
            return

        callback(images, time.time() - start_time, False)

    # Create and start the generation thread.
    thread = Thread(target=worker)
    thread.start()


def gen_error(error: str):
    logger.error(error)
    dpg.hide_item("progress_bar")

    dpg.set_value("status_text", error)
    dpg.show_item("status_text")


def text2img(
    imagen: Text2Img, prepare_UI, callback, progress_callback
) -> list[GeneratedImage]:
    prompt = dpg.get_value("prompt")
    negative_prompt = dpg.get_value("negative_prompt")
    size = dpg.get_values(["width", "height"])
    guidance_scale = dpg.get_value("guidance_scale")
    step_count = dpg.get_value("step_count")
    image_amount = dpg.get_value("image_amount")
    seed = dpg.get_value("seed")

    if seed:
        try:
            # Split and convert seeds into integers.
            if seed.isdigit():
                seed = int(seed)
            else:
                seed = [int(s) for s in re.split(r"[, ]+", seed)]
        except ValueError:
            gen_error("Seed provided are not integers.")
            return

    prepare_UI()

    thread_generate(
        imagen.generate_image,
        callback,
        prompt=prompt,
        negative_prompt=negative_prompt,
        size=size,
        guidance_scale=guidance_scale,
        step_count=step_count,
        seed=seed,
        image_amount=image_amount,
        progress_callback=progress_callback,
    )


def img2img(
    imagen: Img2Img, prepare_UI, callback, progress_callback
) -> list[Img2ImgGeneratedImage]:
    prompt = dpg.get_value("prompt")
    negative_prompt = dpg.get_value("negative_prompt")
    size = dpg.get_values(["width", "height"])
    strength = dpg.get_value("strength")
    guidance_scale = dpg.get_value("guidance_scale")
    step_count = dpg.get_value("step_count")
    image_amount = dpg.get_value("image_amount")
    base_image_path = dpg.get_value("base_image_path")
    seed = dpg.get_value("seed")

    if seed:
        try:
            # Split and convert seeds into integers.
            if seed.isdigit():
                seed = int(seed)
            else:
                seed = [int(s) for s in re.split(r"[, ]+", seed)]
        except ValueError:
            gen_error("Seed provided are not integers.")
            return

    if not os.path.isfile(base_image_path):
        gen_error("Base image does not exist.")
        return

    try:
        base_image = Image.open(base_image_path).resize(size).convert("RGB")
        # To be used when saving image.
        base_image.filename = base_image_path
    except UnidentifiedImageError:
        gen_error("Base image is not an image file.")
        return

    prepare_UI()

    thread_generate(
        imagen.generate_image,
        callback,
        base_image=base_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        strength=strength,
        step_count=step_count,
        seed=seed,
        image_amount=image_amount,
        progress_callback=progress_callback,
    )


def upscale(upscaler, image, callback, **kwargs):
    def worker():
        try:
            upscaled = upscaler.upscale_image(image, **kwargs)
        except Exception as e:
            callback(None, e)
            return

        callback(upscaled, None)

    thread = Thread(target=worker)
    thread.start()
