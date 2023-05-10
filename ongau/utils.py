from diffusers import StableDiffusionImg2ImgPipeline
from PIL.PngImagePlugin import PngInfo
from imagen import GeneratedImage
import os


# edited and modified from https://stackoverflow.com/questions/17984809/how-do-i-create-an-incrementing-filename-in-python
def next_file_number(path_pattern: str):
    """
    Finds the next free file number in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    1
    2
    3

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return b


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
