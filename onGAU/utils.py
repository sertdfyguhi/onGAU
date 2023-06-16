from imagen import GeneratedImage, GeneratedLatents, LatentUpscaledImage

from diffusers import StableDiffusionImg2ImgPipeline
from PIL.PngImagePlugin import PngInfo
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
    info = image_info

    if (info_t := type(info)) not in [GeneratedImage, GeneratedLatents]:
        metadata.add_text(
            "upscaler_type",
            "Latent" if (is_latent := info_t == LatentUpscaledImage) else "RealESRGAN",
        )
        metadata.add_text("upscale_model", info.model_path)

        if is_latent:
            metadata.add_text("u_step_count", str(info.step_count))
            metadata.add_text("u_guidance_scale", str(info.guidance_scale))
        else:
            metadata.add_text("upscale_amount", str(info.upscale_amount))

        info = info.original_image

    scheduler_name = info.scheduler.__name__
    if info.karras_sigmas_used:
        scheduler_name += " Karras"

    if (
        info.scheduler.__name__ == "DPMSolverMultistepScheduler"
        and not info.scheduler_algorithm_type
    ):
        scheduler_name += "++"

    metadata.add_text("model", info.model_path)
    metadata.add_text("prompt", info.prompt)
    metadata.add_text("negative_prompt", info.negative_prompt)
    metadata.add_text("guidance_scale", str(info.guidance_scale))
    metadata.add_text("step_count", str(info.step_count))
    metadata.add_text("pipeline", info.pipeline.__name__)
    metadata.add_text("scheduler", scheduler_name)
    metadata.add_text("seed", str(info.seed))
    metadata.add_text("clip_skip", str(info.clip_skip))
    metadata.add_text("compel_weighting", str(info.compel_weighting))
    metadata.add_text(
        "embeddings",
        ", ".join(
            [
                embedding.replace(",", "\\,").replace(";", "\\;")
                for embedding in info.embeddings
            ]
        ),
    )

    metadata.add_text(
        "loras",
        ";".join([lora.replace(",", "\\,").replace(";", "\\;") for lora in info.loras]),
    )

    if info.pipeline == StableDiffusionImg2ImgPipeline:
        metadata.add_text("strength", str(info.strength))
        metadata.add_text("base_image_path", info.base_image_path)

    image_info.image.save(file_path, pnginfo=metadata)
