from diffusers import SchedulerMixin, DiffusionPipeline
from dataclasses import dataclass
from compel import Compel
from PIL import Image
import copy


@dataclass(frozen=True)
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
    width: int
    height: int


class BaseImagen:
    def __init__(
        self, model: str, device: str, lpw_stable_diffusion: bool = False
    ) -> None:
        self._model = model
        self._device = device
        self._scheduler = None
        self._embedding_models_loaded = []
        self._clip_skip_amount = 0
        self._safety_checker_enabled = False
        self._attention_slicing_enabled = False
        self._vae_slicing_enabled = False
        self._xformers_memory_attention_enabled = False
        self._compel_weighting_enabled = False
        self.set_model(model, lpw_stable_diffusion)

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
    def lpw_stable_diffusion_used(self):
        return self._lpw_stable_diffusion_used

    @classmethod
    def from_class(cls, original):
        c = cls(
            original.model, original.device, original.lpw_stable_diffusion_used
        )  # initialize class

        c.set_clip_skip_amount(original.clip_skip_amount)

        if original.scheduler:
            c.set_scheduler(
                original.scheduler,
                # using private vars here :face_vomiting:
                getattr(original._pipeline.scheduler, "use_karras_sigmas", False),
            )

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

        return c

    def _set_model(
        self,
        model: str,
        pipeline: DiffusionPipeline = DiffusionPipeline,
        scheduler: SchedulerMixin = None,
        use_lpw_stable_diffusion: bool = False,
    ) -> None:
        print(f"loading {model} with {self._device}")

        self._model = model
        self._lpw_stable_diffusion_used = use_lpw_stable_diffusion

        orig_scheduler = None

        if hasattr(self, "_pipeline"):
            orig_scheduler = copy.deepcopy(self._pipeline.scheduler)

            del self._pipeline
            if self._compel_weighting_enabled:
                del self._compel

            if hasattr(self, "_orig_safety_checker"):
                del self._orig_safety_checker

        self._pipeline = pipeline.from_pretrained(
            model,
            custom_pipeline="lpw_stable_diffusion"
            if use_lpw_stable_diffusion
            else None,
        ).to(self._device)

        if scheduler:
            self.set_scheduler(scheduler)  # might implement karras sigmas to this
        else:
            if orig_scheduler:
                self.set_scheduler(
                    orig_scheduler.__class__,
                    getattr(orig_scheduler, "use_karras_sigmas", False),
                )

        self._scheduler = self._pipeline.scheduler.__class__

        # remove progress bar logging
        self._pipeline.set_progress_bar_config(disable=True)

        # for clip skip use
        self._clip_layers = self._pipeline.text_encoder.text_model.encoder.layers
        self.set_clip_skip_amount(self.clip_skip_amount)

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
            self._pipeline.load_textual_inversion(model)

    def load_lpw_stable_diffusion(self):
        self._set_model(self._model, self._pipeline.__class__, self._scheduler, True)

    def load_embedding_model(self, embedding_model_path: str):
        self._embedding_models_loaded.append(embedding_model_path)
        self._pipeline.load_textual_inversion(embedding_model_path)

    def set_clip_skip_amount(self, amount: int = None):
        if amount >= len(self._clip_layers):
            raise ValueError("cannot skip more clip layers")

        self._pipeline.text_encoder.text_model.encoder.layers = (
            self._clip_layers[:-amount] if amount else self._clip_layers
        )

    def set_device(self, device: str):
        self._device = device
        self._pipeline = self._pipeline.to(device)

    def set_scheduler(self, scheduler: SchedulerMixin, use_karras_sigmas: bool = False):
        if use_karras_sigmas:
            self._pipeline.scheduler = scheduler.from_config(
                self._pipeline.scheduler.config, use_karras_sigmas=use_karras_sigmas
            )
        else:
            self._pipeline.scheduler = scheduler.from_config(
                self._pipeline.scheduler.config
            )

        self._scheduler = scheduler
        self._scheduler_uses_karras_sigmas = use_karras_sigmas

    def enable_safety_checker(self):
        if hasattr(self._pipeline, "safety_checker"):
            self._safety_checker_enabled = True
            self._pipeline.safety_checker = self._orig_safety_checker

    def disable_safety_checker(self):
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
                "compel prompt weighting cannot be used when using lpw_stable_diffusion"
            )

        self._compel_weighting_enabled = True
        self._compel = Compel(self._pipeline.tokenizer, self._pipeline.text_encoder)

    def disable_compel_weighting(self):
        self._compel_weighting_enabled = False
