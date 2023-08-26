import dearpygui.dearpygui as dpg
import json
import os

import logger


class ThemeManager:
    def __init__(self, theme_dir: str) -> None:
        """Theme manager."""
        self.theme_dir = theme_dir
        self.current_theme = ""
        self.themes = {}

        for fp in os.listdir(self.theme_dir):
            theme = json.load(open(os.path.join(theme_dir, fp)))

            try:
                if theme["name"] in self.themes:
                    logger.warn(
                        f'Theme "{theme["name"]}" already exists, overriding theme.'
                    )

                self.themes[theme["name"]] = theme["colors"]
            except json.JSONDecodeError as e:
                logger.error(f'Error reading theme "{fp}": {e}')

    def get_themes(self) -> list[str]:
        """Gets a list of available themes."""
        return list(self.themes.keys())

    def load_theme(self, name: str):
        """Loads a theme."""
        if name not in self.themes:
            raise ValueError("Theme doesn't exist.")

        colors = self.themes[name]

        try:
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(
                        dpg.mvThemeCol_FrameBgHovered, colors["item_hover"]
                    )
                    dpg.add_theme_color(
                        dpg.mvThemeCol_ButtonHovered, colors["item_hover"]
                    )
                    dpg.add_theme_color(
                        dpg.mvThemeCol_HeaderHovered, colors["item_hover"]
                    )

                    dpg.add_theme_color(dpg.mvThemeCol_WindowBg, colors["background"])
                    dpg.add_theme_color(
                        dpg.mvThemeCol_TitleBgActive, colors["title_bar"]
                    )
                    dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, colors["menubar"])

                    dpg.add_theme_color(dpg.mvThemeCol_Text, colors["text"])
                    dpg.add_theme_color(
                        dpg.mvThemeCol_TextSelectedBg, colors["text_selected"]
                    )

                    dpg.add_theme_color(dpg.mvThemeCol_PopupBg, colors["popup"])
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBg, colors["item"])
                    dpg.add_theme_color(dpg.mvThemeCol_CheckMark, colors["checkmark"])

                    dpg.add_theme_color(dpg.mvThemeCol_Button, colors["button"])
                    dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, colors["button"])

                with dpg.theme_component(dpg.mvProgressBar):
                    dpg.add_theme_color(
                        dpg.mvThemeCol_PlotHistogram, colors["progress_bar"]
                    )
                    dpg.add_theme_color(
                        dpg.mvThemeCol_Text, colors["progress_bar_text"]
                    )

            dpg.bind_theme(theme)
            self.current_theme = name
        except AttributeError as e:
            logger.error(f'Error while loading theme "{name}": {e}')
