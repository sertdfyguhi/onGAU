from imagen import GeneratedImage

from diffusers import StableDiffusionImg2ImgPipeline
from PIL.PngImagePlugin import PngInfo
import torch
import os


def next_file_number(path_pattern: str, start_from: int = 1):
    """Iteratively find the next file number using a path pattern."""

    i = start_from

    while True:
        if not os.path.exists(path_pattern % i):
            return i

        i += 1


def resize_size_to_fit(
    image_size: tuple[int, int] | list[int, int],
    window_size: tuple[int, int] | list[int, int],
):
    """Resizes an image size to fit within window."""
    img_w, img_h = image_size
    win_w, win_h = window_size

    aspect_ratio = img_w / img_h

    width = min(img_w, win_w)
    height = width // aspect_ratio

    if height > win_h:
        height = win_h
        width = int(height * aspect_ratio)

    return width, height


def append_dir_if_startswith(path: str, dir: str, startswith: str):
    """Checks if a path starts with and if so appends a path to it"""
    return os.path.join(dir, path) if path.startswith(startswith) else path


def save_image(image_info: GeneratedImage, file_path: str):
    """Saves an image using a GeneratedImage object."""
    metadata = PngInfo()
    metadata.add_text("model", image_info.model)
    metadata.add_text("prompt", image_info.prompt)
    metadata.add_text("negative_prompt", image_info.negative_prompt)
    # metadata.add_text("strength", str(image_info.strength))
    metadata.add_text("guidance_scale", str(image_info.guidance_scale))
    metadata.add_text("step_count", str(image_info.step_count))
    metadata.add_text("pipeline", image_info.pipeline.__name__)
    metadata.add_text(
        "scheduler",
        image_info.scheduler.__name__
        + (" Karras" if image_info.karras_sigmas_used else ""),
    )
    metadata.add_text("seed", str(image_info.seed))
    metadata.add_text("clip_skip", str(image_info.clip_skip))
    metadata.add_text(
        "embeddings",
        ", ".join(
            [embedding.replace(",", "\\,") for embedding in image_info.embeddings]
        ),
    )

    BACKSLASH = chr(92)
    metadata.add_text(
        "loras",
        ";".join(
            [
                f'{lora[0].replace(";", BACKSLASH + ";").replace(",", BACKSLASH + ",")}, {lora[1]}'  # sanitize
                for lora in image_info.loras
            ]
        ),
    )

    if image_info.pipeline == StableDiffusionImg2ImgPipeline:
        metadata.add_text("base_image_path", image_info.base_image_path)

    image_info.image.save(file_path, pnginfo=metadata)
