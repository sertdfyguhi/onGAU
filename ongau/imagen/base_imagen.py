import torch

class BaseImagen:
    def __init__(self, model: str, device: str) -> None:
        self._model = model
        self._device = device
        self._attention_slicing_enabled = False
        self._vae_slicing_enabled = False
        self._xformers_memory_attention_enabled = False
        self.set_model(model)

    @property
    def model(self):
        return self._model

    def set_device(self, device: str):
        self._device = device
        self._pipeline = self._pipeline.to(device)

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