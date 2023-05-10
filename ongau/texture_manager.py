from imagen.text2img import GeneratedImage
from PIL.Image import Image
import dearpygui.dearpygui as dpg
import numpy as np


def _convert_PIL_to_DPG_image(pil_image: Image):
    """Converts a Pillow image into a DearPyGui texture image."""
    # Converts the PIL image into a flattened array then turns all color values into floats.
    return np.ravel(np.asarray(pil_image)).astype(np.float32) / 255.0


class TextureManager:
    def __init__(self, texture_reg: str | int) -> None:
        """
        Manages image textures.
        Needed to bypass raw texture memory leak and segfaults from deleting and recreating too many textures.
        """
        self._texture_reg = texture_reg
        self._textures = []
        self._images = None
        self._image_index = 0

    @property
    def images(self):
        return self._images

    def prepare(self, images: list[GeneratedImage]) -> None:
        """Prepare a list of GeneratedImages into the manager."""
        if not self._textures:
            self.clear()

        self._images = images

        for image in images:
            self._textures.append(
                dpg.add_static_texture(
                    width=image.width,
                    height=image.height,
                    default_value=_convert_PIL_to_DPG_image(image.image),
                    parent=self._texture_reg,
                )
            )

    def next(self) -> tuple[str | int, GeneratedImage]:
        """Returns and moves to the next image and texture."""
        if self._image_index == len(self._textures) - 1:
            return

        self._image_index += 1
        return self._textures[self._image_index], self._images[self._image_index]

    def previous(self) -> tuple[str | int, GeneratedImage]:
        """Returns and moves to the previous image and texture."""
        if self._image_index == 0:
            return

        self._image_index -= 1
        return self._textures[self._image_index], self._images[self._image_index]

    def current(self) -> tuple[str | int, GeneratedImage]:
        """Returns the current image."""
        return self._textures[self._image_index], self._images[self._image_index]

    def clear(self):
        """Clear and delete all images and textures from the manager."""
        for tag in self._textures:
            dpg.delete_item(tag)

        self._images = None
        self._textures = []
        self._image_index = 0

    def to_counter_string(self) -> str:
        """Converts image index into a counter string."""
        return f"{self._image_index + 1}/{len(self._textures)}"
