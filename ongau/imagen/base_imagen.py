from diffusers import SchedulerMixin, DiffusionPipeline


class BaseImagen:
    def __init__(
        self,
        model: str,
        device: str,
    ) -> None:
        self._model = model
        self._device = device
        self._safety_checker_enabled = False
        self._attention_slicing_enabled = False
        self._vae_slicing_enabled = False
        self._xformers_memory_attention_enabled = False
        self.set_model(model)

    @property
    def model(self):
        return self._model

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def pipeline(self):
        return self._pipeline.__class__

    def _set_model(
        self, model: str, pipeline=DiffusionPipeline, scheduler: SchedulerMixin = None
    ) -> None:
        print(f"loading {model} with {self._device}")

        self._model = model
        self._pipeline = pipeline.from_pretrained(model).to(self._device)
        if scheduler:
            self.set_scheduler(scheduler)
        self._scheduler = self._pipeline.scheduler.__class__

        # remove progress bar logging
        self._pipeline.set_progress_bar_config(disable=True)

        # make a copy of the safety checker to be able to enable and disable it
        self._orig_safety_checker = self._pipeline.safety_checker

        if not self._safety_checker_enabled:
            self.disable_safety_checker()

        if self._attention_slicing_enabled:
            self.enable_attention_slicing()

        if self._vae_slicing_enabled:
            self.enable_vae_slicing()

        if self._xformers_memory_attention_enabled:
            self.enable_xformers_memory_attention()

    def set_device(self, device: str):
        self._device = device
        self._pipeline = self._pipeline.to(device)

    def set_scheduler(self, scheduler: SchedulerMixin):
        self._scheduler = scheduler
        self._pipeline.scheduler = scheduler.from_config(
            self._pipeline.scheduler.config
        )

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
