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
    """
    Iterative function to resize a specified image size to fit into a specified window size.
    """

    result_image_size = image_size

    while (is_width := result_image_size[0] > window_size[0]) or (
        result_image_size[1] > window_size[1]
    ):
        aspect_ratio = result_image_size[0] / result_image_size[1]

        if is_width:
            result_image_size = [
                window_size[0],
                round(window_size[0] / aspect_ratio),
            ]
        else:
            result_image_size = [round(window_size[1] * aspect_ratio), window_size[1]]

    return result_image_size


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
    if image_info.pipeline == StableDiffusionImg2ImgPipeline:
        metadata.add_text("base_image_path", image_info.base_image_path)

    image_info.image.save(file_path, pnginfo=metadata)
