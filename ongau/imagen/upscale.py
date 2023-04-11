from diffusers import LDMSuperResolutionPipeline
from .text2img import GeneratedImage
from dataclasses import dataclass
from .base import BaseImagen
from PIL.Image import Image
from . import utils


@dataclass(frozen=True)
class LDMUpscaledImage(GeneratedImage):
    original_image: GeneratedImage


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
    ) -> LDMUpscaledImage:
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

        return LDMUpscaledImage(
            self._model,
            utils.convert_PIL_to_DPG_image(image),
            image,
            step_count,
            eta,
            gen_seed,
            *image.size
        )
