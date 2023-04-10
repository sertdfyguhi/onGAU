from imagen.text2img import GeneratedImage
import dearpygui.dearpygui as dpg


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
        self.clear()

        self._images = images

        for image in images:
            self._textures.append(
                dpg.add_static_texture(
                    width=image.width,
                    height=image.height,
                    default_value=image.contents,
                    parent=self._texture_reg,
                )
            )

    def next(self) -> list[str | int, GeneratedImage]:
        if self._image_index == len(self._textures) - 1:
            return

        self._image_index += 1
        return self._textures[self._image_index], self._images[self._image_index]

    def previous(self) -> list[str | int, GeneratedImage]:
        if self._image_index == 0:
            return

        self._image_index -= 1
        return self._textures[self._image_index], self._images[self._image_index]

    def current(self) -> list[str | int, GeneratedImage]:
        return self._textures[self._image_index], self._images[self._image_index]

    def clear(self):
        for tag in self._textures:
            dpg.delete_item(tag)

        self._images = None
        self._textures = []
        self._image_index = 0

    def to_counter_string(self) -> str:
        return f"{self._image_index+1}/{len(self._textures)}"
