from diffusers import StableDiffusionImg2ImgPipeline
from .text2img import GeneratedImage
from .base import BaseImagen
from dataclasses import dataclass
from typing import Callable
from PIL.Image import Image
from . import utils


@dataclass(frozen=True)
class Img2ImgGeneratedImage(GeneratedImage):
    base_image: Image


class SDImg2Img(BaseImagen):
    def __init__(self, model: str, device: str) -> None:
        super().__init__(model, device)

    def set_model(self, model: str):
        self._set_model(model, StableDiffusionImg2ImgPipeline)

    def generate_image(
        self,
        base_image: Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.8,
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
                    # strength=strength,
                    num_inference_steps=step_count,
                    guidance_scale=guidance_scale,
                    callback=progress_callback,
                ).images[0]
                for i in range(image_amount)
            ]
        else:
            images = self._pipeline(
                image=base_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generators,
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
                Img2ImgGeneratedImage(
                    self._model,
                    utils.convert_PIL_to_DPG_image(image),
                    image,
                    prompt,
                    negative_prompt,
                    strength,
                    guidance_scale,
                    step_count,
                    seeds[i],
                    self.pipeline,
                    self.scheduler,
                    *image.size,
                    base_image
                )
            )

        return result
