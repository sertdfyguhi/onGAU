from diffusers import LDMSuperResolutionPipeline
from .base_imagen import BaseImagen
from dataclasses import dataclass
from PIL.Image import Image
from . import utils
import numpy as np

@dataclass(frozen=True)
class UpscaledImage:
    model: str
    contents: np.ndarray
    image: Image
    step_count: int
    eta: int
    seed: int
    width: int
    height: int


class ImageUpscalerLDM(BaseImagen):
    def __init__(self, model: str, device: str) -> None:
        super().__init__(model, device)

    def set_model(self, model: str) -> None:
        print(f"loading {model} with {self._device}")

        self._model = model
        self._pipeline = LDMSuperResolutionPipeline.from_pretrained(model).to(self._device)

        if self._attention_slicing_enabled:
            self.enable_attention_slicing()

        if self._vae_slicing_enabled:
            self.enable_vae_slicing()

        if self._xformers_memory_attention_enabled:
            self.enable_xformers_memory_attention()

        # remove progress bar logging
        self._pipeline.set_progress_bar_config(disable=True)

    def upscale_image(
        self,
        image: Image,
        step_count: int = 25,
        eta: int = 1,
        seed: int = None,
        # progress_callback: Callable = None
    ) -> UpscaledImage:
        generator, gen_seed = utils.create_torch_generator(seed, self._device)
        image = (
            self._pipeline(
                image=image,
                generator=generator,
                num_inference_steps=step_count,
                eta=eta,
                # callback=progress_callback
            )
            .images[0]
            .convert("RGBA")
        )

        return UpscaledImage(
            self._model,
            utils.convert_PIL_to_DPG_image(image),
            image,
            step_count,
            eta,
            gen_seed,
            *image.size
        )
