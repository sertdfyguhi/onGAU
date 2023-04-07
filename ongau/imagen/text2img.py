from diffusers import DiffusionPipeline
from .base_imagen import BaseImagen
from dataclasses import dataclass
from PIL.Image import Image
from typing import Callable
from . import utils
import numpy as np


@dataclass(frozen=True)
class GeneratedImage:
    model: str
    contents: np.ndarray
    image: Image
    prompt: str
    negative_prompt: str
    strength: int
    guidance_scale: int
    step_count: int
    seed: int
    width: int
    height: int


class ImageGenerator(BaseImagen):
    def __init__(self, model: str, device: str) -> None:
        self._safety_checker_enabled = False
        super().__init__(model, device)

    def set_model(self, model: str) -> None:
        print(f"loading {model} with {self._device}")

        self._model = model
        self._pipeline = DiffusionPipeline.from_pretrained(model).to(self._device)

        if not self._safety_checker_enabled:
            self.disable_safety_checker()

        if self._attention_slicing_enabled:
            self.enable_attention_slicing()

        if self._vae_slicing_enabled:
            self.enable_vae_slicing()

        if self._xformers_memory_attention_enabled:
            self.enable_xformers_memory_attention()

        # remove progress bar logging
        self._pipeline.set_progress_bar_config(disable=True)

        # make a copy of the safety checker to be able to enable and disable it
        self._orig_safety_checker = self._pipeline.safety_checker

    def enable_safety_checker(self):
        self._safety_checker_enabled = True
        self._pipeline.safety_checker = self._orig_safety_checker

    def disable_safety_checker(self):
        if self._pipeline.safety_checker:
            self._safety_checker_enabled = False
            self._pipeline.safety_checker = lambda images, clip_input: (images, False)

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        size: tuple[int, int] | list[int, int] = (512, 512),
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        step_count: int = 25,
        image_amount: int = 1,
        seed: int = None,
        progress_callback: Callable = None,
    ) -> list[GeneratedImage]:
        generators, seeds = utils.create_torch_generator(seed, self._device, image_amount)
        images = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generators,
            width=size[0],
            height=size[1],
            num_inference_steps=step_count,
            # strength=strength, # remove due to weird keyword argument error
            guidance_scale=guidance_scale,
            num_images_per_prompt=image_amount,
            callback=progress_callback
        ).images

        result = []

        for i, image in enumerate(images):
            image = image.convert('RGBA')

            result.append(
                GeneratedImage(
                    self._model,
                    utils.convert_PIL_to_DPG_image(image),
                    image,
                    prompt,
                    negative_prompt,
                    strength,
                    guidance_scale,
                    step_count,
                    seeds[i],
                    *image.size
                )
            )

        return result