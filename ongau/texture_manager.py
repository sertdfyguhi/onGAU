from imagen.text2img import GeneratedImage
import dearpygui.dearpygui as dpg
from PIL.Image import Image
import numpy as np


def _convert_PIL_to_DPG_image(pil_image: Image):
    # create np array and flatten
    array = np.ravel(np.array(pil_image))
    # convert to float array
    array = array.astype("float32")
    # turn rgba values into floating point numbers
    array = array / 255.0

    return array


class TextureManager:
    def __init__(self, texture_reg: str | int) -> None:
        self._texture_reg = texture_reg
        self._textures = []
        self._images = None
        self._image_index = 0

    @property
    def images(self):
        return self._images

    def prepare(self, images: list[GeneratedImage]) -> None:
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
        if self._image_index == len(self._textures) - 1:
            return

        self._image_index += 1
        return self._textures[self._image_index], self._images[self._image_index]

    def previous(self) -> tuple[str | int, GeneratedImage]:
        if self._image_index == 0:
            return

        self._image_index -= 1
        return self._textures[self._image_index], self._images[self._image_index]

    def current(self) -> tuple[str | int, GeneratedImage]:
        return self._textures[self._image_index], self._images[self._image_index]

    def clear(self):
        for tag in self._textures:
            dpg.delete_item(tag)

        self._images = None
        self._textures = []
        self._image_index = 0

    def to_counter_string(self) -> str:
        return f"{self._image_index+1}/{len(self._textures)}"
