from imagen import SCHEDULERS
from windows.utils import *
from callbacks import *
import windows
import logger
import config

import dearpygui.dearpygui as dpg
import pyperclip
import atexit


# Register UI font.
with dpg.font_registry():
    default_font = dpg.add_font(FONT, config.FONT_SIZE)

# Register key shortcuts.
with dpg.handler_registry():
    dpg.add_key_press_handler(
        dpg.mvKey_Left, callback=lambda: switch_image_callback("previous")
    )
    dpg.add_key_press_handler(
        dpg.mvKey_Right, callback=lambda: switch_image_callback("next")
    )

# Create dialog box for loading settings from an image file.
with dpg.window(
    label="Load settings from image",
    tag="image_load_dialog",
    pos=CENTER,
    modal=True,
):
    dpg.add_input_text(
        label="Image Path",
        tag="image_path_input",
        width=config.ITEM_WIDTH,
    )
    dpg.add_checkbox(
        label="Ignore Model", tag="ignore_model_checkbox", default_value=True
    )
    dpg.add_checkbox(
        label="Ignore Pipeline", tag="ignore_pipeline_checkbox", default_value=False
    )

    with dpg.group(horizontal=True):
        dpg.add_button(
            label="Load",
            width=config.ITEM_WIDTH / 2 + 35,
            callback=load_from_image_callback,
        )
        dpg.add_button(
            label="Cancel",
            width=config.ITEM_WIDTH / 2 + 35,
            callback=lambda: dpg.hide_item("image_load_dialog"),
        )

# Create dialog box for loading settings from an image file.
with dpg.window(
    label="Save settings",
    tag="save_settings_dialog",
    pos=CENTER,
    modal=True,
):
    dpg.add_input_text(
        label="Save Name",
        tag="save_name_input",
        width=config.ITEM_WIDTH - config.FONT_SIZE * 5,
    )
    dpg.add_checkbox(
        label="Include Model", tag="include_model_checkbox", default_value=True
    )

    with dpg.group(horizontal=True):
        dpg.add_button(
            label="Save",
            width=config.ITEM_WIDTH / 2 - 5,
            callback=save_settings_callback,
        )
        dpg.add_button(
            label="Cancel",
            width=config.ITEM_WIDTH / 2 - 5,
            callback=lambda: dpg.hide_item("save_settings_dialog"),
        )

# Create dialog box for loading settings from an image file.
with dpg.window(
    label="Delete save",
    tag="delete_save_dialog",
    pos=CENTER,
    modal=True,
):
    dpg.add_combo(
        label="Saves",
        tag="delete_save_input",
    )
    update_delete_save_input()

    with dpg.group(horizontal=True):
        dpg.add_button(
            label="Delete",
            width=config.ITEM_WIDTH / 2 - 5,
            callback=delete_save_callback,
        )
        dpg.add_button(
            label="Cancel",
            width=config.ITEM_WIDTH / 2 - 5,
            callback=lambda: dpg.hide_item("delete_save_dialog"),
        )

windows.upscale.init()
windows.merge.init()

# Main window.
with dpg.window(tag="window"):
    with dpg.menu_bar():
        with dpg.menu(label="Saves", tag="saves_menu"):
            for name in settings_manager.settings:
                saves_tags[name] = dpg.add_menu_item(
                    label=name, callback=(lambda n: lambda: load_save_callback(n))(name)
                )

            dpg.add_menu_item(
                label="Delete Save...",
                tag="delete_save_button",
                callback=lambda: toggle_item("delete_save_dialog"),
            )
            dpg.add_menu_item(
                label="Save Current Settings...",
                callback=lambda: toggle_item("save_settings_dialog"),
            )

        with dpg.menu(label="File"):
            dpg.add_menu_item(
                label="Load Settings from Image",
                callback=lambda: toggle_item("image_load_dialog"),
            )

        dpg.add_menu_item(
            label="Merge",
            callback=windows.merge.toggle,
        )

    dpg.add_input_text(
        label="Model",
        default_value=imagen.model_path,
        width=config.ITEM_WIDTH,
        tag="model",
    )
    add_tooltip(
        "The path to a Stable Diffusion model to use. (HuggingFace model or local model)"
    )

    dpg.add_input_text(
        label="Prompt",
        default_value=user_settings["prompt"],
        width=config.ITEM_WIDTH,
        tag="prompt",
    )
    add_tooltip("The instructions of the generated image.")

    dpg.add_input_text(
        label="Negative Prompt",
        default_value=user_settings["negative_prompt"],
        width=config.ITEM_WIDTH,
        tag="negative_prompt",
    )
    add_tooltip("The instructions of what to remove of the generated image.")

    dpg.add_input_int(
        label="Width",
        default_value=int(user_settings["width"]),
        min_value=1,
        width=config.ITEM_WIDTH,
        callback=image_size_calc_callback,
        tag="width",
    )
    add_tooltip("The image width of the output image.")

    dpg.add_input_int(
        label="Height",
        default_value=int(user_settings["height"]),
        min_value=1,
        width=config.ITEM_WIDTH,
        callback=image_size_calc_callback,
        tag="height",
    )
    add_tooltip("The image height of the output image.")

    with dpg.group(tag="strength_group", show=user_settings["pipeline"] == "SDImg2Img"):
        dpg.add_input_float(
            label="Denoising Strength",
            default_value=0.80,
            min_value=0.00,
            max_value=1.00,
            format="%.2f",
            width=config.ITEM_WIDTH,
            tag="strength",
        )
        add_tooltip("The amount of noise added to the base image in img2img.")

    dpg.add_input_float(
        label="Guidance Scale",
        default_value=float(user_settings["guidance_scale"]),
        min_value=0.0,
        max_value=50.0,
        format="%.1f",
        width=config.ITEM_WIDTH,
        tag="guidance_scale",
    )
    add_tooltip("How closely SD should follow your prompt.")

    dpg.add_input_int(
        label="Step Count",
        default_value=int(user_settings["step_count"]),
        min_value=1,
        max_value=500,
        width=config.ITEM_WIDTH,
        tag="step_count",
    )
    add_tooltip(
        "The number of iterations that SD runs over the image. Higher number usually gets you a better image."
    )

    dpg.add_input_int(
        label="Amount of Images",
        default_value=int(user_settings["image_amount"]),
        min_value=1,
        max_value=100,
        width=config.ITEM_WIDTH,
        tag="image_amount",
    )
    add_tooltip("The amount of images to generate.")

    dpg.add_input_text(
        label="Seed",
        default_value=user_settings["seed"],
        width=config.ITEM_WIDTH,
        tag="seed",
    )
    add_tooltip(
        "The number to initialize the generation. Leaving it empty chooses it randomly."
    )

    # file dialog to be used
    # with dpg.group(horizontal=True):
    #     btn = dpg.add_button(label="Choose...")
    # Group to hide and show tooltip as it breaks if you only hide the parent.
    with dpg.group(
        tag="base_image_group", show=user_settings["pipeline"] == "SDImg2Img"
    ):
        dpg.add_input_text(
            # readonly=True,
            label="Base Image Path",
            default_value=user_settings["base_image_path"],
            width=config.ITEM_WIDTH,
            tag="base_image_path",
            callback=base_image_path_callback,
        )
        add_tooltip("The path of the starting image to use in img2img.")

    dpg.add_button(
        label="Advanced Configuration", callback=lambda: toggle_item("advanced_config")
    )

    with dpg.group(tag="advanced_config", indent=7, show=False):
        dpg.add_combo(
            label="Pipeline",
            items=["Text2Img", "SDImg2Img"],
            default_value=imagen.__class__.__name__,
            width=config.ITEM_WIDTH,
            callback=change_pipeline_callback,
            tag="pipeline",
        )
        add_tooltip("The pipeline to use.")

        dpg.add_combo(
            label="Scheduler",
            items=SCHEDULERS,
            default_value=user_settings["scheduler"] or imagen.scheduler.__name__,
            width=config.ITEM_WIDTH,
            callback=lambda _, scheduler: load_scheduler(scheduler),
        )
        add_tooltip("The scheduling / sampling method to use.")

        dpg.add_input_int(
            label="Clip Skip",
            default_value=int(user_settings["clip_skip"]),
            width=config.ITEM_WIDTH,
            tag="clip_skip",
        )
        add_tooltip("The amount CLIP layers to skip.")

        dpg.add_checkbox(
            label="Disable Safety Checker",
            tag="safety_checker",
            default_value=not imagen.safety_checker_enabled,
            callback=lambda tag, value: checkbox_callback(tag, not value),
        )
        add_tooltip(
            "Disables check for NSFW images. If enabled, images detected as NSFW are replaced with a black image."
        )

        dpg.add_checkbox(
            label="Enable Attention Slicing",
            tag="attention_slicing",
            default_value=imagen.attention_slicing_enabled,
            callback=checkbox_callback,
        )
        add_tooltip(
            "Slices the computation into multiple steps. Increases performance on Apple Silicon chips."
        )

        dpg.add_checkbox(
            label="Enable Vae Slicing",
            tag="vae_slicing",
            default_value=imagen.vae_slicing_enabled,
            callback=checkbox_callback,
        )
        add_tooltip("Slices VAE decoding into multiple steps.")

        dpg.add_checkbox(
            label="Enable xFormers Memory Efficient Attention",
            tag="xformers_memory_attention",
            default_value=imagen.xformers_memory_attention_enabled,
            callback=toggle_xformers_callback,
        )

        dpg.add_checkbox(
            label="Enable Model CPU Offload",
            tag="model_cpu_offload",
            default_value=imagen.model_cpu_offload_enabled,
            callback=checkbox_callback,
        )
        add_tooltip(
            "Offloads the model onto the CPU. Reduces memory usage while keeping performance at best."
        )

        dpg.add_checkbox(
            label="Enable Compel Prompt Weighting",
            tag="compel_weighting",
            default_value=imagen.compel_weighting_enabled,
            callback=checkbox_callback,
        )
        add_tooltip(
            "Use compel prompt weighting. + to increase weight and - to decrease."
        )

        dpg.add_checkbox(
            label="Enable LPW Stable Diffusion Pipeline",
            tag="lpwsd_pipeline",
            default_value=imagen.lpw_stable_diffusion_used,
            callback=lpwsd_callback,
        )
        add_tooltip(
            "Use LPWSD pipeline. Adds prompt weighting as seen in A1111's webui and long prompts."
        )

        dpg.add_button(
            label="Save model weights",
            tag="save_model",
            callback=save_model_callback,
        )
        add_tooltip("Used to convert a ckpt or safetensors file into diffusers format.")

    dpg.add_button(
        label="Generate Image", tag="generate_btn", callback=generate_image_callback
    )

    with dpg.group(tag="gen_status_group", horizontal=True, show=False):
        dpg.add_button(
            label="Interrupt Generation",
            callback=interrupt_callback,
            tag="interrupt_btn",
        )
        dpg.add_button(
            label="Kill Generation",
            callback=kill_gen_callback,
            tag="kill_gen_btn",
        )

    # change tag name to smth better
    with dpg.group(tag="output_button_group", horizontal=True, show=False):
        dpg.add_button(
            label="Upscale",
            callback=windows.upscale.toggle,
        )
        dpg.add_button(
            label="Save Image", tag="save_button", callback=save_image_callback
        )

    with dpg.group(horizontal=True, tag="info_group", show=False):
        dpg.add_text(tag="info_text")
        dpg.add_button(
            label="Copy Seed",
            callback=lambda: pyperclip.copy(texture_manager.current()[1].seed),
        )

    dpg.add_text(tag="status_text")

    dpg.add_progress_bar(
        overlay="0%", tag="progress_bar", width=config.ITEM_WIDTH, show=False
    )

    with dpg.group(pos=(440, 27), show=False, tag="output_image_group"):
        # Needed to be able to add output image on a new line before this group.
        with dpg.group(tag="output_image_group2", horizontal=True):
            dpg.add_button(label="<", tag="previous", callback=switch_image_callback)
            dpg.add_button(label=">", tag="next", callback=switch_image_callback)
            dpg.add_text(tag="output_image_index")

            with dpg.group(tag="output_image_btns", horizontal=True):
                dpg.add_button(
                    label="Use In Img2Img",
                    tag="use_in_img2img_btn",
                    callback=use_in_img2img_callback,
                )
                dpg.add_button(
                    label="Reuse Seed",
                    callback=reuse_seed_callback,
                )

    dpg.bind_font(default_font)

dpg.set_primary_window("window", True)

if __name__ == "__main__":
    logger.success("Starting GUI...")

    dpg.set_exit_callback(lambda: settings_manager.save_settings("main"))
    atexit.register(dpg.destroy_context)

    dpg.create_viewport(
        title=config.WINDOW_TITLE,
        width=config.WINDOW_SIZE[0],
        height=config.WINDOW_SIZE[1],
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
