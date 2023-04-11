from diffusers import DiffusionPipeline, SchedulerMixin
from .base import BaseImagen
from dataclasses import dataclass
from PIL.Image import Image
from typing import Callable
from . import utils
import numpy as np


@dataclass(frozen=True)
class GeneratedImage:
    model: str
    image: Image
    prompt: str
    negative_prompt: str
    # strength: int
    guidance_scale: int
    step_count: int
    seed: int
    pipeline: DiffusionPipeline
    scheduler: SchedulerMixin
    width: int
    height: int


# stable diffusion model
class Text2Img(BaseImagen):
    def __init__(self, model: str, device: str) -> None:
        super().__init__(model, device)

    def set_model(self, model: str):
        self._set_model(model, DiffusionPipeline)

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        size: tuple[int, int] | list[int, int] = (512, 512),
        # strength: float = 0.8,
        guidance_scale: float = 8.0,
        step_count: int = 25,
        image_amount: int = 1,
        seed: int | list[int] = None,
        progress_callback: Callable = None,
    ) -> list[GeneratedImage]:
        generators, seeds = utils.create_torch_generator(
            seed, self._device, image_amount
        )

        if (type(seed) == list or image_amount > 1) and self._device == "mps":
            images = [
                self._pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    generator=generators[i],
                    width=size[0],
                    height=size[1],
                    # strength=strength,
                    num_inference_steps=step_count,
                    guidance_scale=guidance_scale,
                    callback=progress_callback,
                ).images[0]
                for i in range(image_amount)
            ]
        else:
            images = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generators,
                width=size[0],
                height=size[1],
                num_inference_steps=step_count,
                # strength=strength,
                guidance_scale=guidance_scale,
                num_images_per_prompt=image_amount,
                callback=progress_callback,
            ).images

        result = []

        for i, image in enumerate(images):
            image = image.convert("RGBA")

            result.append(
                GeneratedImage(
                    model=self._model,
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    # strength=strength,
                    guidance_scale=guidance_scale,
                    step_count=step_count,
                    seed=seeds[i],
                    pipeline=self.pipeline,
                    scheduler=self.scheduler,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

        return result
