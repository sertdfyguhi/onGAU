from diffusers import DiffusionPipeline
from dataclasses import dataclass
from PIL.Image import Image
from typing import Callable
import numpy as np
import torch


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


class ImageGeneration:
    def __init__(self, model: str, device: str) -> None:
        self._device = device
        self._safety_checker_enabled = False
        self._attention_slicing_enabled = False
        self.set_model(model)

    @property
    def model(self):
        return self._model

    def set_model(self, model: str) -> None:
        print(f"loading {model} with {self._device}")

        self._model = model
        self._pipeline = DiffusionPipeline.from_pretrained(model).to(self._device)

        if not self._safety_checker_enabled:
            self.disable_safety_checker()

        if self._attention_slicing_enabled:
            self.enable_attention_slicing()

        # remove progress bar logging
        self._pipeline.set_progress_bar_config(disable=True)

        # make a copy of the safety checker to be able to enable and disable it
        self._orig_safety_checker = self._pipeline.safety_checker

    def set_device(self, device: str):
        self._device = device
        self._pipeline = self._pipeline.to(device)

    def enable_safety_checker(self):
        self._safety_checker_enabled = True
        self._pipeline.safety_checker = self._orig_safety_checker

    def disable_safety_checker(self):
        if self._pipeline.safety_checker:
            self._safety_checker_enabled = False
            self._pipeline.safety_checker = lambda images, clip_input: (images, False)

    def enable_attention_slicing(self):
        self._attention_slicing_enabled = True
        self._pipeline.enable_attention_slicing()

    def disable_attention_slicing(self):
        self._attention_slicing_enabled = False
        self._pipeline.disable_attention_slicing()

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        size: tuple[int, int] | list[int, int] = (512, 512),
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        step_count: int = 25,
        seed: int = None,
        progress_callback: Callable = None,
    ) -> GeneratedImage:
        generator = torch.Generator(device=self._device)

        if seed:
            generator = generator.manual_seed(seed)
        else:
            generator.seed()

        generation_seed = generator.initial_seed()
        image = (
            self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                width=size[0],
                height=size[1],
                num_inference_steps=step_count,
                # strength=strength, # remove due to weird keyword argument error
                guidance_scale=guidance_scale,
                callback=progress_callback
            )
            .images[0]
            .convert("RGBA")
        )

        # create np array and flatten
        array = np.ravel(np.array(image))
        # convert to float array
        array = array.astype("float32")
        # turn rgba values into floating point numbers
        array = array / 255.0

        return GeneratedImage(
            self._model,
            array,
            image,
            prompt,
            negative_prompt,
            strength,
            guidance_scale,
            step_count,
            generation_seed,
            *image.size
        )
