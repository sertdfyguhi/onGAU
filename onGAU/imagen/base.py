from . import utils

from diffusers import SchedulerMixin, DiffusionPipeline
from huggingface_hub.utils import HFValidationError
from dataclasses import dataclass
from compel import Compel
from PIL import Image
import torch
import os


@dataclass
class GeneratedImage:
    model: str
    image: Image
    prompt: str
    negative_prompt: str
    guidance_scale: int
    step_count: int
    seed: int
    pipeline: DiffusionPipeline
    scheduler: SchedulerMixin
    karras_sigmas_used: bool
    scheduler_algorithm_type: str
    compel_weighting: bool
    clip_skip: int
    loras: list[str]
    embeddings: list[str]
    width: int
    height: int


@dataclass
class GeneratedLatents:
    model: str
    latents: torch.Tensor
    prompt: str
    negative_prompt: str
    guidance_scale: int
    step_count: int
    seeds: list[int]
    pipeline: DiffusionPipeline
    scheduler: SchedulerMixin
    karras_sigmas_used: bool
    scheduler_algorithm_type: str
    compel_weighting: bool
    clip_skip: int
    loras: list[str]
    embeddings: list[str]
    width: int
    height: int


class BaseImagen:
    def __init__(
        self, model: str, device: str, use_lpw_stable_diffusion: bool = False
    ) -> None:
        """Base class for all imagen classes."""
        self._model = model
        self._device = device
        self._scheduler = None
        self._scheduler_algorithm_type = None
        self._loras_loaded = []
        self._embedding_models_loaded = []
        self._clip_skip_amount = 0
        self._karras_sigmas_used = False
        self._safety_checker_enabled = False
        self._attention_slicing_enabled = False
        self._vae_slicing_enabled = False
        self._xformers_memory_attention_enabled = False
        self._compel_weighting_enabled = False
        self.set_model(model, use_lpw_stable_diffusion)

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def scheduler_algorithm_type(self):
        return self._scheduler_algorithm_type

    @property
    def pipeline(self):
        return self._pipeline.__class__

    @property
    def clip_skip_amount(self):
        return self._clip_skip_amount

    @property
    def safety_checker_enabled(self):
        return self._safety_checker_enabled

    @property
    def attention_slicing_enabled(self):
        return self._attention_slicing_enabled

    @property
    def vae_slicing_enabled(self):
        return self._vae_slicing_enabled

    @property
    def xformers_memory_attention_enabled(self):
        return self._xformers_memory_attention_enabled

    @property
    def compel_weighting_enabled(self):
        return self._compel_weighting_enabled

    @property
    def embedding_models_loaded(self):
        return self._embedding_models_loaded

    @property
    def loras_loaded(self):
        return self._loras_loaded

    @property
    def lpw_stable_diffusion_used(self):
        return self._lpw_stable_diffusion_used

    @property
    def karras_sigmas_used(self):
        return self._karras_sigmas_used

    @classmethod
    def from_class(cls, original):
        """Create a new imagen object from another imagen object."""
        c = cls(
            original.model, original.device, original.lpw_stable_diffusion_used
        )  # initialize class

        c.set_clip_skip_amount(original.clip_skip_amount)

        if original.scheduler:
            c.set_scheduler(original.scheduler, original.karras_sigmas_used)

        if not original.safety_checker_enabled:
            c.disable_safety_checker()

        if original.attention_slicing_enabled:
            c.enable_attention_slicing()

        if original.vae_slicing_enabled:
            c.enable_vae_slicing()

        if original.xformers_memory_attention_enabled:
            c.enable_xformers_memory_attention()

        if original.compel_weighting_enabled:
            c.enable_compel_weighting()

        for model in original.embedding_models_loaded:
            c.load_embedding_model(model)

        for lora in original.loras_loaded:
            c.load_lora(*lora)

        return c

    def _set_model(
        self,
        model: str,
        pipeline: DiffusionPipeline = DiffusionPipeline,
        scheduler: SchedulerMixin = None,
        use_lpw_stable_diffusion: bool = False,
    ) -> None:
        """Base function to set the model of the pipeline."""
        self._model = model

        if use_lpw_stable_diffusion:
            if self._compel_weighting_enabled:
                raise RuntimeError(
                    "Compel prompt weighting cannot be used when using LPWSD pipeline."
                )
            elif model.endswith((".ckpt", ".safetensors")):
                raise ValueError(
                    "LPWSD pipeline is not compatible with a .ckpt or .safetensors file."
                )

        orig_scheduler = None

        if hasattr(self, "_pipeline"):
            orig_scheduler = self._pipeline.scheduler.__class__

            del self._pipeline
            if self._compel_weighting_enabled:
                del self._compel

            if hasattr(self, "_orig_safety_checker"):
                del self._orig_safety_checker

        try:
            self._pipeline = (
                pipeline.from_ckpt
                if model.endswith((".ckpt", ".safetensors"))
                else pipeline.from_pretrained
            )(
                model,
                custom_pipeline="lpw_stable_diffusion"
                if use_lpw_stable_diffusion
                else None,
            )
        except HFValidationError:
            raise FileNotFoundError(f"{model} does not exist.")

        self._lpw_stable_diffusion_used = use_lpw_stable_diffusion

        if scheduler:
            self.set_scheduler(scheduler)  # might implement karras sigmas to this
        elif orig_scheduler:
            self.set_scheduler(
                orig_scheduler,
                self._karras_sigmas_used,
            )

        self._scheduler = self._pipeline.scheduler.__class__

        # remove progress bar logging
        self._pipeline.set_progress_bar_config(disable=True)

        # for clip skip use
        self._clip_layers = self._pipeline.text_encoder.text_model.encoder.layers
        self.set_clip_skip_amount(self._clip_skip_amount, force=True)

        # make a copy of the safety checker to be able to enable and disable it
        if hasattr(self._pipeline, "safety_checker"):
            self._orig_safety_checker = self._pipeline.safety_checker

        if not self._safety_checker_enabled:
            self.disable_safety_checker()

        if self._attention_slicing_enabled:
            self.enable_attention_slicing()

        if self._vae_slicing_enabled:
            self.enable_vae_slicing()

        if self._xformers_memory_attention_enabled:
            self.enable_xformers_memory_attention()

        if self._compel_weighting_enabled:
            self.enable_compel_weighting()

        for model in self._embedding_models_loaded:
            self.load_embedding_model(model)

        for lora in self._loras_loaded:
            self.load_lora(*lora)

        self.set_device(self._device)

    def load_lpw_stable_diffusion(self):
        """Load Long Prompt Weighting Stable Diffusion pipeline."""
        self._set_model(self._model, self._pipeline.__class__, self._scheduler, True)

    def load_embedding_model(self, embedding_model_path: str):
        """Load a textual inversion model."""
        if embedding_model_path in self._embedding_models_loaded:
            return

        try:
            self._pipeline.load_textual_inversion(
                embedding_model_path,
                os.path.basename(embedding_model_path).split(".")[0],
            )
        except ValueError:  # when tokenizer already has that token
            return

        self._embedding_models_loaded.append(embedding_model_path)

    def set_clip_skip_amount(self, amount: int = None, force: bool = False):
        if amount >= len(self._clip_layers):
            raise ValueError(
                "Clip skip higher than amount of clip layers, no clip skip has been applied."
            )

        if not force and amount == self._clip_skip_amount:
            return

        self._pipeline.text_encoder.text_model.encoder.layers = (
            self._clip_layers[:-amount] if amount else self._clip_layers
        )

        self._clip_skip_amount = amount

    def load_lora(self, lora_path: str, weight: float = 0.75):
        """Load a .safetensors lora."""
        if lora_path in [l[0] for l in self._loras_loaded]:
            return

        self._pipeline = utils.load_lora(
            self._pipeline,
            lora_path,
            self._device,
            weight,
        )
        self._loras_loaded.append((lora_path, weight))

    def set_device(self, device: str):
        """Change device of pipeline."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch, "has_mps", False):
                device = "mps"
            else:
                device = "cpu"

        self._device = device
        self._pipeline = self._pipeline.to(device)

    # TODO: Add DPM++ SDE Karras.
    def set_scheduler(
        self,
        scheduler: SchedulerMixin,
        use_karras_sigmas: bool = False,
        algorithm_type: str | None = None,
    ):
        """Change scheduler of pipeline."""
        # TODO: Set scheduler internal variable instead of reinstating when using same scheduler
        kwargs = {
            key: value
            for key, value in {
                "use_karras_sigmas": use_karras_sigmas,
                "algorithm_type": algorithm_type,
            }.items()
            if value
        }

        print(kwargs)

        self._pipeline.scheduler = scheduler.from_config(
            self._pipeline.scheduler.config, **kwargs
        )

        self._scheduler = scheduler
        self._karras_sigmas_used = use_karras_sigmas
        self._scheduler_algorithm_type = algorithm_type

    def save_weights(self, dir_path: str):
        """Save model weights in diffusers format in directory path."""
        orig_clip_skip = self._clip_skip_amount
        self.set_clip_skip_amount(0)

        self._pipeline.save_pretrained(dir_path)

        self.set_clip_skip_amount(orig_clip_skip)

    def enable_safety_checker(self):
        """Enable the safety checker."""
        if hasattr(self._pipeline, "safety_checker"):
            self._safety_checker_enabled = True
            self._pipeline.safety_checker = self._orig_safety_checker

    def disable_safety_checker(self):
        """Disable the safety checker."""
        if hasattr(self._pipeline, "safety_checker") and self._pipeline.safety_checker:
            self._safety_checker_enabled = False
            self._pipeline.safety_checker = lambda images, clip_input: (images, False)

    def enable_attention_slicing(self):
        self._attention_slicing_enabled = True
        self._pipeline.enable_attention_slicing()

    def disable_attention_slicing(self):
        self._attention_slicing_enabled = False
        self._pipeline.disable_attention_slicing()

    def enable_vae_slicing(self):
        self._vae_slicing_enabled = True
        self._pipeline.enable_vae_slicing()

    def disable_vae_slicing(self):
        self._vae_slicing_enabled = False
        self._pipeline.disable_vae_slicing()

    def enable_xformers_memory_attention(self):
        self._xformers_memory_attention_enabled = True
        self._pipeline.enable_xformers_memory_efficient_attention()

    def disable_xformers_memory_attention(self):
        self._xformers_memory_attention_enabled = False
        self._pipeline.disable_xformers_memory_efficient_attention()

    def enable_compel_weighting(self):
        if self._lpw_stable_diffusion_used:
            raise RuntimeError(
                "Compel prompt weighting cannot be used when using LPWSD pipeline."
            )

        self._compel_weighting_enabled = True
        self._compel = Compel(self._pipeline.tokenizer, self._pipeline.text_encoder)

    def disable_compel_weighting(self):
        self._compel_weighting_enabled = False
