from diffusers import DiffusionPipeline, SchedulerMixin
from .base import BaseImagen
from dataclasses import dataclass
from PIL.Image import Image
from typing import Callable
from . import utils
import torch


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
    def set_model(self, model: str, lpw_stable_diffusion: bool = False):
        self._set_model(
            model, DiffusionPipeline, lpw_stable_diffusion=lpw_stable_diffusion
        )

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
            seed,
            "cpu"
            if self._lpw_stable_diffusion_used and self._device == "mps"
            else self._device,
            image_amount,
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

        kwargs = {
            "prompt": temp_prompt,
            "negative_prompt": temp_negative_prompt,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "width": size[0],
            "height": size[1],
            "num_inference_steps": step_count,
            # strength=strength,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": image_amount,
            "callback": progress_callback,
        }

        if self._lpw_stable_diffusion_used:
            del kwargs["prompt_embeds"], kwargs["negative_prompt_embeds"]

        if self._lpw_stable_diffusion_used or (
            self._device == "mps" and len(seeds) > 1
        ):
            images = [
                self._pipeline(**kwargs, generator=generators[i]).images[0]
                for i in range(image_amount)
            ]
        else:
            images = self._pipeline(**kwargs, generator=generators).images

        result = []

        for i, image in enumerate(images):
            result.append(
                GeneratedImage(
                    model=self._model,
                    image=image.convert("RGBA"),
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

        del prompt_embeds, negative_prompt_embeds

        return result
