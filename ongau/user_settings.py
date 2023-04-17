import dearpygui.dearpygui as dpg
import configparser
import config


class UserSettings:
    def __init__(self, settings_file: str):
        self._settings_file = settings_file
        self._config = configparser.ConfigParser()
        self._config.read(settings_file)

    def get_user_settings(self):
        user_settings = {}

        for op in [
            "model",
            "prompt",
            "negative_prompt",
            "seed",
            "pipeline",
            "guidance_scale",
            "step_count",
            "image_amount",
            "width",
            "height",
        ]:
            user_settings[op] = self._config.get(
                "user_settings", op, fallback=getattr(config, "DEFAULT_" + op.upper())
            )

        for op in [
            "scheduler",
            "safety_checker",
            "attention_slicing",
            "vae_slicing",
            "xformers_memory_attention",
            "compel_weighting",
        ]:
            user_settings[op] = self._config.get("user_settings", op, fallback=None)

        user_settings["base_image_path"] = self._config.get(
            "user_settings", "base_image_path", fallback=""
        )

        user_settings["clip_skip"] = self._config.get(
            "user_settings", "clip_skip", fallback=0
        )

        return user_settings

    def save_user_settings(self):
        if not self._config.has_section("user_settings"):
            self._config.add_section("user_settings")

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
            self._config.set("user_settings", op, dpg.get_value(op))

        # options that do need to be converted to a string
        for op in [
            "guidance_scale",
            "step_count",
            "image_amount",
            "width",
            "height",
            "clip_skip",
            "safety_checker",
            "attention_slicing",
            "vae_slicing",
            "xformers_memory_attention",
            "compel_weighting",
        ]:
            self._config.set(
                "user_settings",
                op,
                str(dpg.get_value(op)),
            )

        with open(self._settings_file, "w") as config_file:
            self._config.write(config_file)
