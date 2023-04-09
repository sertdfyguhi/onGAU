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

    def set_model(self, model: str):
        self._set_model(model, LDMSuperResolutionPipeline)

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
