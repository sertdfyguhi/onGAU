import configparser
import dearpygui.dearpygui as dpg

class ConfigReader:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('User_Inputs.ini')

    def get_default_prompt(self):
        return self.config.get('Default', 'prompt')

    def get_default_negative(self):
        return self.config.get('Default', 'negative')

    def get_default_scale(self):
        return float(self.config.get('Default', 'scale'))

    def get_default_step(self):
        return int(self.config.get('Default', 'step'))

    def get_default_numimage(self):
        return int(self.config.get('Default', 'numimage'))

    def get_default_seed(self):
        return self.config.get('Default', 'seed')

    def get_default_pipeline(self):
        return self.config.get('Default', 'pipeline')

class UserSettings:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('User_Inputs.ini')
    def save_user_settings(self):
        user_prompt = dpg.get_value("prompt")
        self.config.set(section='Default', option='prompt', value=str(user_prompt))
        user_negative_prompt = dpg.get_value("negative_prompt")
        self.config.set(section='Default', option='negative', value=str(user_negative_prompt))
        user_pipeline = dpg.get_value("pipeline")
        self.config.set(section='Default', option='pipeline', value=str(user_pipeline))
        user_guidance_scale = dpg.get_value("guidance_scale")
        self.config.set(section='Default', option='scale', value=str(user_guidance_scale))
        user_step_count = dpg.get_value("step_count")
        self.config.set(section='Default', option='step', value=str(user_step_count))
        user_image_amount = dpg.get_value("image_amount")
        self.config.set(section='Default', option='numimage', value=str(user_image_amount))
        user_seed = dpg.get_value("seed")
        self.config.set(section='Default', option='seed', value=str(user_seed))
        with open("User_Inputs.ini", "w") as config_file:
            self.config.write(config_file)