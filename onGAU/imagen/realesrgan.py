from .base import GeneratedImage

# Hack to fix a changed import in torchvision 0.17+, which otherwise breaks
# basicsr; see https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985
try:
    import torchvision.transforms.functional_tensor  # type: ignore
except ImportError:
    try:
        import torchvision.transforms.functional as functional
        import sys

        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass  # shrug...

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch
import os


@dataclass
class RealESRGANUpscaledImage:
    model_path: str
    upscale_amount: int
    width: int
    height: int
    image: Image.Image
    original_image: GeneratedImage


class RealESRGAN:
    def __init__(self, model: str, device: str) -> None:
        """RealESRGAN model."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch, "has_mps", False):
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.set_model(model)

    def set_model(
        self,
        model_path: str,
        half_precision: bool = False,
    ):
        """Set the RealESRGAN model path."""
        if not os.path.isfile(model_path):
            raise ValueError("Model path does not exist.")

        model_name = os.path.basename(model_path).split(".")[0]

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
            raise ValueError("Failed to determine RealESRGAN model type.")

        self._pipeline = RealESRGANer(
            scale=scale,
            device=self.device,
            model_path=model_path,
            model=model,
            half=half_precision,
        )
        self.model_path = model_path

    def set_tile_size(self, tile_size: int):
        """Set the amount of tiles to create."""
        self._pipeline.tile_size = tile_size

    def upscale_image(
        self, generated_image: GeneratedImage | Image.Image, upscale: int | None = None
    ) -> RealESRGANUpscaledImage:
        """Upscale a GeneratedImage object using ESRGAN."""
        try:
            output, _ = self._pipeline.enhance(
                np.array(
                    generated_image
                    if isinstance(generated_image, Image.Image)
                    else generated_image.image
                ),
                outscale=upscale,
            )
        except RuntimeError:
            raise RuntimeError(
                "Too much memory allocated. Enable tiling to reduce memory usage (not implemented yet)."
            )

        height, width, _ = output.shape

        return RealESRGANUpscaledImage(
            model_path=self.model_path,
            upscale_amount=upscale,
            width=width,
            height=height,
            image=Image.fromarray(output),
            original_image=generated_image,
        )
