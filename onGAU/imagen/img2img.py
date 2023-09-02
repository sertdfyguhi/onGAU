from .base import BaseImagen, GeneratedImage, GeneratedLatents
from . import utils

from diffusers import StableDiffusionImg2ImgPipeline
from dataclasses import dataclass, asdict
from typing import Callable
from PIL.Image import Image
import torch
import time


@dataclass
class Img2ImgGeneratedImage(GeneratedImage):
    strength: float
    base_image_path: str


@dataclass
class Img2ImgGeneratedLatents(GeneratedLatents):
    strength: float
    base_image_path: str


class SDImg2Img(BaseImagen):
    def set_model(
        self,
        model: str,
        precision: str = "fp32",
        use_lpw_stable_diffusion: bool = False,
    ):
        self._set_model(
            model,
            precision=precision,
            pipeline=StableDiffusionImg2ImgPipeline,
            use_lpw_stable_diffusion=use_lpw_stable_diffusion,
        )

    def convert_latent_to_image(
        self, latents: Img2ImgGeneratedLatents
    ) -> GeneratedImage:
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
            Img2ImgGeneratedImage(
                image=image.convert("RGBA"),
                seed=latents.seeds[i],
                **dict_latents,
            )
            for i, image in enumerate(images)
        ]

    def generate_image(
        self,
        base_image: Image,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 8.0,
        strength: float = 0.8,
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

        if self.compel_weighting_enabled:
            temp_prompt = temp_negative_prompt = None
            prompt_embeds = self._compel.build_conditioning_tensor(prompt)
            negative_prompt_embeds = self._compel.build_conditioning_tensor(
                negative_prompt
            )

        # Use for callback.
        out_image_kwargs = {
            "base_image_path": base_image.filename,
            "model_path": self.model_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "step_count": step_count,
            "pipeline": self.pipeline,
            "scheduler": self.scheduler,
            "compel_weighting": self.compel_weighting_enabled,
            "clip_skip": self.clip_skip_amount,
            "loras": self.loras_loaded,
            "embeddings": self.embedding_models_loaded,
            "width": base_image.size[0],
            "height": base_image.size[1],
        }

        last_step_time = time.time()
        step_times = []

        def callback_wrapper(step: int, _, latents):
            nonlocal last_step_time, step_times

            step_times.append(time.time() - last_step_time)

            progress_callback(
                step,
                step_count,
                step_times,
                Img2ImgGeneratedLatents(
                    **out_image_kwargs, seeds=seeds, latents=latents
                ),
            )

            last_step_time = time.time()

        kwargs = {
            "image": base_image,
            "prompt": temp_prompt,
            "negative_prompt": temp_negative_prompt,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "num_inference_steps": step_count,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "num_images_per_prompt": image_amount,
            "callback": callback_wrapper if progress_callback else None,
        }

        if self.lpw_stable_diffusion_used:
            # lpwsd pipeline does not accept prompt embeds
            del kwargs["prompt_embeds"], kwargs["negative_prompt_embeds"]
            kwargs["max_embeddings_multiples"] = self.max_embeddings_multiples

        if self.device == "mps" and len(seeds) > 1:
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
                Img2ImgGeneratedImage(
                    **out_image_kwargs,
                    image=image.convert("RGBA"),
                    seed=seeds[i],
                )
            )

        del prompt_embeds, negative_prompt_embeds

        return result
