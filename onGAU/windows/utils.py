import dearpygui.dearpygui as dpg
import config

CENTER = (config.WINDOW_SIZE[0] / 2, config.WINDOW_SIZE[1] / 2)


def add_tooltip(text: str):
    """Adds a tooltip to the last added widget."""
    dpg.add_text(
        text,
        parent=dpg.add_tooltip(dpg.last_item()),
    )


def toggle_item(tag: str | int):
    """Toggle visibility of an item."""
    if dpg.is_item_shown(tag):
        dpg.hide_item(tag)
    else:
        dpg.show_item(tag)
