from diffusers import StableDiffusionImg2ImgPipeline
from .text2img import GeneratedImage
from dataclasses import dataclass
from .base import BaseImagen
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

        prompt_embeds = negative_prompt_embeds = None
        temp_prompt = prompt
        temp_negative_prompt = negative_prompt

        if self._compel_weighting_enabled:
            temp_prompt = temp_negative_prompt = None
            prompt_embeds = self._compel.build_conditioning_tensor(prompt)
            negative_prompt_embeds = self._compel.build_conditioning_tensor(
                negative_prompt
            )

        if self._device == "mps" and len(seeds) > 1:
            images = [
                self._pipeline(
                    image=base_image,
                    prompt=temp_prompt,
                    negative_prompt=temp_negative_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
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
                prompt=temp_prompt,
                negative_prompt=temp_negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
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
                    base_image=base_image,
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
