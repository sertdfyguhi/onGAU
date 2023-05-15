from imagen import SDImg2Img, Img2ImgGeneratedImage, Text2Img, GeneratedImage
import logger

from PIL import UnidentifiedImageError, Image
from threading import Thread
import dearpygui.dearpygui as dpg
import re
import os


def _generate(func, callback, **kwargs):
    logger.info("Starting generation...")

    def worker():
        try:
            # TODO: Handle generation errors.
            images = func(**kwargs)
        except RuntimeError as e:  #
            callback(e.args[1], True)
            return

        callback(images, False)

    # Create and start the generation thread.
    thread = Thread(target=worker)
    thread.start()


def _error(error: str):
    logger.error(error)
    dpg.hide_item("progress_bar")

    dpg.set_value("status_text", error)
    dpg.show_item("status_text")


def text2img(imagen: Text2Img, callback, progress_callback) -> list[GeneratedImage]:
    prompt = dpg.get_value("prompt")
    negative_prompt = dpg.get_value("negative_prompt")
    size = dpg.get_values(["width", "height"])
    # strength = dpg.get_value("strength")
    guidance_scale = dpg.get_value("guidance_scale")
    step_count = dpg.get_value("step_count")
    image_amount = dpg.get_value("image_amount")
    seed = dpg.get_value("seed")
    if seed:
        try:
            # Split and convert seeds into integers.
            seed = (
                int(seed)
                if seed.isdigit()
                else [int(s) for s in re.split(r"[, ]+", seed)]
            )
        except ValueError:
            _error("Seeds provided are not integers.")
            return

    _generate(
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
    imagen: SDImg2Img, callback, progress_callback
) -> list[Img2ImgGeneratedImage]:
    prompt = dpg.get_value("prompt")
    negative_prompt = dpg.get_value("negative_prompt")
    size = dpg.get_values(["width", "height"])
    # strength = dpg.get_value("strength")
    guidance_scale = dpg.get_value("guidance_scale")
    step_count = dpg.get_value("step_count")
    image_amount = dpg.get_value("image_amount")
    base_image_path = dpg.get_value("base_image_path")
    seed = dpg.get_value("seed")
    if seed:
        try:
            # Split and convert seeds into integers.
            seed = (
                int(seed)
                if seed.isdigit()
                else [int(s) for s in re.split(r"[, ]+", seed)]
            )
        except ValueError:
            _error("Seeds provided are not integers.")
            return

    if not os.path.isfile(base_image_path):
        _error("Base image path does not exist.")
        return

    try:
        base_image = Image.open(base_image_path).resize(size).convert("RGB")
        # To be used when saving image.
        base_image.filename = base_image_path
    except UnidentifiedImageError:
        _error("Base image path is not an image file.")
        return

    _generate(
        imagen.generate_image,
        callback,
        base_image=base_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        step_count=step_count,
        seed=seed,
        image_amount=image_amount,
        progress_callback=progress_callback,
    )
