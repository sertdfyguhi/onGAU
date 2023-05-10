from diffusers import StableDiffusionPipeline
from .base import BaseImagen, GeneratedImage
from typing import Callable
from . import utils


# stable diffusion model
class Text2Img(BaseImagen):
    def set_model(self, model: str, use_lpw_stable_diffusion: bool = False):
        self._set_model(
            model,
            StableDiffusionPipeline,
            use_lpw_stable_diffusion=use_lpw_stable_diffusion,
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
            else self._device,  # bug in lpwsd pipeline that causes it to break when using mps generators
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
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": image_amount,
            "callback": progress_callback,
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
                    karras_sigmas_used=self._karras_sigmas_used,
                    clip_skip=self._clip_skip_amount,
                    loras=self._loras_loaded,
                    embeddings=self._embedding_models_loaded,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

        del prompt_embeds, negative_prompt_embeds

        return result
