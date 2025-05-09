from .base import BaseImagen, GeneratedImage, GeneratedLatents
from . import utils

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from dataclasses import asdict
from typing import Callable
from PIL import Image
import torch
import time


# stable diffusion model
class Text2Img(BaseImagen):
    def set_model(
        self,
        model: str,
        precision: str = "fp32",
        use_lpw_stable_diffusion: bool = False,
        sdxl: bool = False,
    ):
        self._set_model(
            model,
            precision=precision,
            pipeline=StableDiffusionXLPipeline if sdxl else StableDiffusionPipeline,
            use_lpw_stable_diffusion=use_lpw_stable_diffusion,
        )

    def convert_latent_to_image(self, latents: GeneratedLatents) -> GeneratedImage:
        """Convert a GeneratedLatent object into a GeneratedImage object."""
        # Convert latent space image into PIL image.
        images = []

        for latent in (
            latents.latents if type(latents.latents) == list else [latents.latents]
        ):
            image_latents = 1 / 0.18215 * latent
            with torch.no_grad():
                image = self._pipeline.vae.decode(image_latents).sample

            image = (image / 2 + 0.5).clamp(0, 1).squeeze()
            image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
            images.append(Image.fromarray(image))

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

        prompt_embeds = negative_prompt_embeds = pooled_prompt_embeds = (
            negative_pooled_prompt_embeds
        ) = None
        temp_prompt = prompt
        temp_negative_prompt = negative_prompt

        if self.compel_weighting_enabled:
            temp_prompt = temp_negative_prompt = None

            if self.sdxl:
                prompt_embeds, pooled_prompt_embeds = self._compel(prompt)
                negative_prompt_embeds, negative_pooled_prompt_embeds = self._compel(
                    negative_prompt
                )
            else:
                prompt_embeds = self._compel.build_conditioning_tensor(prompt)
                negative_prompt_embeds = self._compel.build_conditioning_tensor(
                    negative_prompt
                )

        # Use for callback.
        out_image_kwargs = {
            "model_path": self.model_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "step_count": step_count,
            "pipeline": self.pipeline,
            "scheduler": self.scheduler,
            "compel_weighting": self.compel_weighting_enabled,
            "clip_skip": self.clip_skip_amount,
            "loras": self.loras_loaded,
            "embeddings": self.embedding_models_loaded,
            "width": size[0],
            "height": size[1],
        }

        last_step_time = time.time()
        start_time = time.time()

        def callback_wrapper(pipe, step: int, _, kwargs):
            nonlocal last_step_time

            now = time.time()

            progress_callback(
                step + 1,
                step_count,
                now - last_step_time,
                now - start_time,
                GeneratedLatents(
                    **out_image_kwargs, seeds=seeds, latents=kwargs["latents"]
                ),
            )

            last_step_time = now
            return kwargs

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
            "callback_on_step_end": callback_wrapper if progress_callback else None,
        }

        if self.lpw_stable_diffusion_used:
            # lpwsd pipeline does not accept prompt embeds
            del kwargs["prompt_embeds"], kwargs["negative_prompt_embeds"]
            kwargs["max_embeddings_multiples"] = self.max_embeddings_multiples
            kwargs["callback"] = kwargs.pop("callback_on_step_end")

        if self.compel_weighting_enabled and self.sdxl:
            kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
            kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

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
                GeneratedImage(
                    **out_image_kwargs,
                    image=image.convert("RGBA"),
                    seed=seeds[i],
                )
            )

        del (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        return result
