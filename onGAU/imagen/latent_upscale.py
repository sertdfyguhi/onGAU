from .base import GeneratedImage

from diffusers import StableDiffusionLatentUpscalePipeline
from huggingface_hub.utils import HFValidationError
from dataclasses import dataclass
from PIL import Image
import torch


@dataclass
class LatentUpscaledImage:
    model_path: str
    guidance_scale: str
    step_count: str
    width: int
    height: int
    image: Image.Image
    original_image: GeneratedImage


class LatentUpscaler:
    def __init__(self, model: str, device: str) -> None:
        self.model_path = model
        self.device = device

        self.safety_checker_enabled = False
        self.attention_slicing_enabled = False
        self.vae_slicing_enabled = False
        self.model_cpu_offload_enabled = False
        self.xformers_memory_attention_enabled = False

        self.set_model(model)

    def set_model(self, model: str):
        try:
            self._pipeline = StableDiffusionLatentUpscalePipeline.from_pretrained(model)
        except HFValidationError:
            raise FileNotFoundError(f"{model} does not exist.")

        self._pipeline.set_progress_bar_config(disable=True)

        # make a copy of the safety checker to be able to enable and disable it
        if hasattr(self._pipeline, "safety_checker"):
            self._orig_safety_checker = self._pipeline.safety_checker

        if not self.safety_checker_enabled:
            self.disable_safety_checker()

        if self.attention_slicing_enabled:
            self.enable_attention_slicing()

        if self.vae_slicing_enabled:
            self.enable_vae_slicing()

        if self.model_cpu_offload_enabled:
            self.enable_model_cpu_offload()

        if self.xformers_memory_attention_enabled:
            self.enable_xformers_memory_attention()

        self.model_path = model
        self.set_device(self.device)

    def set_device(self, device: str):
        """Change device of pipeline."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch, "has_mps", False):
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self._pipeline = self._pipeline.to(device)

    def disable_safety_checker(self):
        """Disable the safety checker."""
        if hasattr(self._pipeline, "safety_checker") and self._pipeline.safety_checker:
            self.safety_checker_enabled = False
            self._pipeline.safety_checker = lambda images, clip_input: (images, False)

    def enable_attention_slicing(self):
        self.attention_slicing_enabled = True
        self._pipeline.enable_attention_slicing()

    def disable_attention_slicing(self):
        self.attention_slicing_enabled = False
        self._pipeline.disable_attention_slicing()

    def enable_vae_slicing(self):
        self.vae_slicing_enabled = True
        self._pipeline.enable_vae_slicing()

    def disable_vae_slicing(self):
        self.vae_slicing_enabled = False
        self._pipeline.disable_vae_slicing()

    def enable_xformers_memory_attention(self):
        self.xformers_memory_attention_enabled = True
        self._pipeline.enable_xformers_memory_efficient_attention()

    def disable_xformers_memory_attention(self):
        self.xformers_memory_attention_enabled = False
        self._pipeline.disable_xformers_memory_efficient_attention()

    def enable_model_cpu_offload(self):
        self.model_cpu_offload_enabled = True
        self._pipeline.enable_model_cpu_offload()

    def disable_model_cpu_offload(self):
        self.model_cpu_offload_enabled = False

        # Reinstantiates the model since you cannot disable model cpu offload.
        self.set_model(self.model_path)

    def upscale_image(
        self,
        image: GeneratedImage,
        guidance_scale: float = 8.0,
        step_count: int = 25,
        progress_callback=None,
    ):
        new = self._pipeline(
            prompt=image.prompt,
            negative_prompt=image.negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=step_count,
            image=image.image,
            callback=progress_callback,
        ).images[0]

        return LatentUpscaledImage(
            model_path=self.model_path,
            guidance_scale=guidance_scale,
            step_count=step_count,
            width=new.size[0],
            height=new.size[1],
            image=new,
            original_image=image,
        )
