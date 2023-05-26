from .base import BaseImagen, GeneratedImage, GeneratedLatents
from . import utils

from diffusers import StableDiffusionPipeline
from dataclasses import asdict
from typing import Callable
import torch
import time


# stable diffusion model
class Text2Img(BaseImagen):
    def set_model(self, model: str, use_lpw_stable_diffusion: bool = False):
        self._set_model(
            model,
            StableDiffusionPipeline,
            use_lpw_stable_diffusion=use_lpw_stable_diffusion,
        )

    def convert_latent_to_image(self, latents: GeneratedLatents) -> GeneratedImage:
        """Convert a GeneratedLatent object into a GeneratedImage object."""
        # Convert latent space image into PIL image.
        with torch.no_grad():
            images = self._pipeline.numpy_to_pil(
                self._pipeline.decode_latents(latents.latents)
            )

        dict_latents = asdict(latents)

        # Remove latents and seeds since GeneratedImage object does not contain a latents/seeds value.
        del dict_latents["latents"]
        del dict_latents["seeds"]

        return [
            GeneratedImage(
                image=image.convert("RGBA"),
                seed=latents.seeds[i],
                **dict_latents,
            )
            for i, image in enumerate(images)
        ]

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
            "cpu",
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

        # Use for callback.
        out_image_kwargs = {
            "model": self._model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "step_count": step_count,
            "pipeline": self.pipeline,
            "scheduler": self.scheduler,
            "karras_sigmas_used": self._karras_sigmas_used,
            "scheduler_algorithm_type": self._scheduler_algorithm_type,
            "compel_weighting": self._compel_weighting_enabled,
            "clip_skip": self._clip_skip_amount,
            "loras": self._loras_loaded,
            "embeddings": self._embedding_models_loaded,
            "width": size[0],
            "height": size[1],
        }

        last_step_time = time.time()

        def callback_wrapper(step: int, _, latents):
            nonlocal last_step_time

            progress_callback(
                step,
                step_count,
                time.time() - last_step_time,
                GeneratedLatents(**out_image_kwargs, seeds=seeds, latents=latents),
            )

            last_step_time = time.time()

        kwargs = {
            "prompt": temp_prompt,
            "negative_prompt": temp_negative_prompt,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "width": size[0],
            "height": size[1],
            "num_inference_steps": step_count,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": image_amount,
            "callback": callback_wrapper,
        }

        if self._lpw_stable_diffusion_used:
            # lpwsd pipeline does not accept prompt embeds
            del kwargs["prompt_embeds"], kwargs["negative_prompt_embeds"]
            kwargs["max_embeddings_multiples"] = 6

        # lpwsd pipeline does not work with a list of generators
        if self._lpw_stable_diffusion_used or (
            self._device == "mps" and len(seeds) > 1
        ):
            kwargs["num_images_per_prompt"] = 1
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
                    **out_image_kwargs,
                    image=image.convert("RGBA"),
                    seed=seeds[i],
                )
            )

        del prompt_embeds, negative_prompt_embeds

        return result
