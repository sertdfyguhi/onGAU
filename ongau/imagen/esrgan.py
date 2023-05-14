from .base import GeneratedImage

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch
import cv2
import os


@dataclass
class ESRGANUpscaledImage:
    model: str
    upscale_amount: int
    width: int
    height: int
    seed: int
    image: Image.Image
    original_image: GeneratedImage


def _convert_cv2_to_PIL(cv2_image: cv2.Mat):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def _convert_PIL_to_cv2(pil_image: Image.Image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


class ESRGAN:
    def __init__(self, esrgan_model: str, device: str) -> None:
        """ESRGAN model."""
        self._model = esrgan_model

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch, "has_mps", False):
                device = "mps"
            else:
                device = "cpu"

        self._device = device
        self.set_model(esrgan_model)

    def set_model(
        self,
        esrgan_model: str,
        tile_size: int = 0,
        half_precision: bool = False,
    ):
        """Set the ESRGAN model path."""
        if not os.path.isfile(esrgan_model):
            raise ValueError("Model path does not exist.")

        model_name = os.path.basename(esrgan_model).split(".")[0]

        if model_name in [
            "RealESRGAN_x4plus",
            "RealESRNet_x4plus",
        ]:  # x4 RRDBNet model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            scale = 4
        elif (
            model_name == "RealESRGAN_x4plus_anime_6B"
        ):  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4,
            )
            scale = 4
        elif model_name == "RealESRGAN_x2plus":  # x2 RRDBNet model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            scale = 2
        else:
            raise ValueError("Failed to find ESRGAN model type.")

        self._esrgan = RealESRGANer(
            scale=scale,
            device=self._device,
            model_path=esrgan_model,
            model=model,
            tile=tile_size,
            half=half_precision,
        )
        self._model = esrgan_model

    def set_tile_size(self, tile_size: int):
        """Set the amount of tiles to create."""
        self._esrgan.tile_size = tile_size

    def upscale_image(
        self, generated_image: GeneratedImage, upscale: int
    ) -> ESRGANUpscaledImage:
        """Upscale a GeneratedImage object using ESRGAN."""
        cv2_image = _convert_PIL_to_cv2(generated_image.image)
        output, _ = self._esrgan.enhance(cv2_image, outscale=upscale)
        height, width, _ = output.shape

        return ESRGANUpscaledImage(
            model=self._model,
            upscale_amount=upscale,
            width=width,
            height=height,
            seed=generated_image.seed,
            image=_convert_cv2_to_PIL(output).convert("RGBA"),
            original_image=generated_image,
        )
