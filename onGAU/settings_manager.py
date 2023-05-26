from configparser import ConfigParser
import dearpygui.dearpygui as dpg
import config
import os


class SettingsManager:
    def __init__(self, settings_file: str):
        if not os.path.isfile(settings_file):
            open(settings_file, "w").close()

        self._settings_file = settings_file
        self._config = ConfigParser()
        self._config.read(settings_file)

        # Backwards compatibility
        if self._config.has_section("user_settings"):
            # Rename "user_settings" section to "main" section.
            # self._config["main"] = self._config.pop("user_settings")
            self._config["main"] = self._config["user_settings"]
            self._config.remove_section("user_settings")

    @property
    def settings(self):
        return [key for key in self._config.keys() if key not in ["DEFAULT", "main"]]

    def delete_settings(self, section: str):
        self._config.remove_section(section)

    def get_settings(self, section: str, full: bool = False):
        if full:
            return self._config[section]

        settings = {
            op: self._config.get(
                section,
                op,
                fallback=getattr(config, f"DEFAULT_{op.upper()}"),
            )
            for op in [
                "model",
                "prompt",
                "negative_prompt",
                "seed",
                "pipeline",
                "guidance_scale",
                "step_count",
                "upscale_amount",
                "image_amount",
                "width",
                "height",
                "lpwsd_pipeline",
            ]
        }

        for op in [
            "scheduler",
            "attention_slicing",
            "vae_slicing",
            "xformers_memory_attention",
            "compel_weighting",
        ]:
            settings[op] = self._config.get(section, op, fallback=None)

        settings["safety_checker"] = self._config.get(
            section, "safety_checker", fallback="True"
        )

        settings["base_image_path"] = self._config.get(
            section, "base_image_path", fallback=""
        )

        settings["clip_skip"] = self._config.get(section, "clip_skip", fallback=0)

        return settings

    def save_settings(self, section: str, ignore_keys: list = []):
        if not self._config.has_section(section):
            self._config.add_section(section)

        # options that dont need to be converted to strings
        for op in [
            "model",
            "prompt",
            "negative_prompt",
            "seed",
            "pipeline",
            "scheduler",
            "base_image_path",
        ]:
            # print(op, dpg.get_value(op))
            self._config[section][op] = dpg.get_value(op)

        # options that do need to be converted to a string
        for op in [
            "guidance_scale",
            "step_count",
            "image_amount",
            "width",
            "height",
            "clip_skip",
            "upscale_amount",
            "safety_checker",
            "attention_slicing",
            "vae_slicing",
            "xformers_memory_attention",
            "compel_weighting",
            "lpwsd_pipeline",
        ]:
            self._config[section][op] = str(dpg.get_value(op))

        for key in ignore_keys:
            del self._config[section][key]

        with open(self._settings_file, "w") as config_file:
            self._config.write(config_file)
