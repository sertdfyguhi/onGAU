from callbacks import *
import logger
import config

import dearpygui.dearpygui as dpg
import pyperclip
import atexit

CENTER = (config.WINDOW_SIZE[0] / 2, config.WINDOW_SIZE[1] / 2)

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
        width=config.ITEM_WIDTH - config.FONT_SIZE * 5,
    )

    with dpg.group(horizontal=True):
        dpg.add_button(
            label="Load",
            width=config.ITEM_WIDTH / 2 - 5,
            callback=load_from_image_callback,
        )
        dpg.add_button(
            label="Cancel",
            width=config.ITEM_WIDTH / 2 - 5,
            callback=lambda: dpg.hide_item("image_load_dialog"),
        )

# Create dialog box for loading settings from an image file.
with dpg.window(
    label="Save settings",
    tag="save_settings_dialog",
    pos=(config.WINDOW_SIZE[0] / 2, config.WINDOW_SIZE[1] / 2),
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
    pos=(config.WINDOW_SIZE[0] / 2, config.WINDOW_SIZE[1] / 2),
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

with dpg.window(label="Upscale", show=False, tag="upscale_window", pos=CENTER):
    dpg.add_input_int(
        label="Upscale Amount",
        default_value=int(user_settings["upscale_amount"]),
        min_value=1,
        width=config.ITEM_WIDTH * 0.75,
        tag="upscale_amount",
    )
    dpg.add_text(
        "The amount to upscale the image.",
        parent=dpg.add_tooltip("upscale_amount"),
    )

    dpg.add_button(
        label="Upscale Image",
        tag="upscale_button",
        callback=upscale_image_callback,
    )

# Main window.
with dpg.window(tag="window"):
    with dpg.menu_bar():
        with dpg.menu(label="Saves", tag="saves_menu"):
            for name in settings_manager.settings:
                saves_tags[name] = dpg.add_menu_item(
                    label=name, callback=lambda: load_save_callback(name)
                )

            dpg.add_menu_item(
                label="Delete Save...",
                tag="delete_save_button",
                callback=lambda: dpg.show_item("delete_save_dialog"),
            )
            dpg.add_menu_item(
                label="Save Current Settings...",
                callback=lambda: dpg.show_item("save_settings_dialog"),
            )

        with dpg.menu(label="File"):
            dpg.add_menu_item(
                label="Load Settings from Image",
                callback=lambda: dpg.show_item("image_load_dialog"),
            )

    dpg.add_input_text(
        label="Model",
        default_value=imagen.model_path,
        width=config.ITEM_WIDTH,
        tag="model",
    )
    dpg.add_text(
        "The path to a Stable Diffusion model to use. (HuggingFace model or local model)",
        parent=dpg.add_tooltip("model"),
    )

    dpg.add_input_text(
        label="Prompt",
        default_value=user_settings["prompt"],
        width=config.ITEM_WIDTH,
        tag="prompt",
    )
    dpg.add_text(
        "The instructions of the generated image.",
        parent=dpg.add_tooltip("prompt"),
    )

    dpg.add_input_text(
        label="Negative Prompt",
        default_value=user_settings["negative_prompt"],
        width=config.ITEM_WIDTH,
        tag="negative_prompt",
    )
    dpg.add_text(
        "The instructions of what to remove of the generated image.",
        parent=dpg.add_tooltip("negative_prompt"),
    )

    dpg.add_input_int(
        label="Width",
        default_value=int(user_settings["width"]),
        min_value=1,
        width=config.ITEM_WIDTH,
        callback=image_size_calc_callback,
        tag="width",
    )
    dpg.add_text(
        "The image width of the output image.",
        parent=dpg.add_tooltip("width"),
    )

    dpg.add_input_int(
        label="Height",
        default_value=int(user_settings["height"]),
        min_value=1,
        width=config.ITEM_WIDTH,
        callback=image_size_calc_callback,
        tag="height",
    )
    dpg.add_text(
        "The image height of the output image.",
        parent=dpg.add_tooltip("height"),
    )

    with dpg.group(tag="strength_group", show=user_settings["pipeline"] == "SDImg2Img"):
        dpg.add_input_float(
            label="Denoising Strength",
            default_value=0.80,
            min_value=0.00,
            max_value=1.00,
            format="%.01f",
            width=config.ITEM_WIDTH,
            tag="strength",
        )
        dpg.add_text(
            "The amount of noise added to the base image in img2img.",
            parent=dpg.add_tooltip("strength"),
        )

    dpg.add_input_float(
        label="Guidance Scale",
        default_value=float(user_settings["guidance_scale"]),
        min_value=0.0,
        max_value=50.0,
        format="%.1f",
        width=config.ITEM_WIDTH,
        tag="guidance_scale",
    )
    dpg.add_text(
        "How closely SD should follow your prompt.",
        parent=dpg.add_tooltip("guidance_scale"),
    )

    dpg.add_input_int(
        label="Step Count",
        default_value=int(user_settings["step_count"]),
        min_value=1,
        max_value=500,
        width=config.ITEM_WIDTH,
        tag="step_count",
    )
    dpg.add_text(
        "The number of iterations that SD runs over the image. Higher number usually gets you a better image.",
        parent=dpg.add_tooltip("step_count"),
    )

    dpg.add_input_int(
        label="Amount of Images",
        default_value=int(user_settings["image_amount"]),
        min_value=1,
        max_value=100,
        width=config.ITEM_WIDTH,
        tag="image_amount",
    )
    dpg.add_text(
        "The amount of images to generate.",
        parent=dpg.add_tooltip("image_amount"),
    )

    dpg.add_input_text(
        label="Seed",
        default_value=user_settings["seed"],
        width=config.ITEM_WIDTH,
        tag="seed",
    )
    dpg.add_text(
        "The number to initialize the generation. Leaving it empty chooses it randomly.",
        parent=dpg.add_tooltip("seed"),
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
        dpg.add_text(
            "The path of the starting image to use in img2img.",
            parent=dpg.add_tooltip("base_image_path"),
        )

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
        dpg.add_text(
            "The pipeline to use.",
            parent=dpg.add_tooltip("pipeline"),
        )

        dpg.add_combo(
            label="Scheduler",
            items=config.SCHEDULERS,
            default_value=user_settings["scheduler"] or imagen.scheduler.__name__,
            width=config.ITEM_WIDTH,
            tag="scheduler",
        )
        dpg.add_text(
            "The sampling method to use.",
            parent=dpg.add_tooltip("scheduler"),
        )

        dpg.add_input_int(
            label="Clip Skip",
            default_value=int(user_settings["clip_skip"]),
            width=config.ITEM_WIDTH,
            tag="clip_skip",
        )
        dpg.add_text(
            "The amount CLIP layers to skip.",
            parent=dpg.add_tooltip("clip_skip"),
        )

        dpg.add_checkbox(
            label="Disable Safety Checker",
            tag="safety_checker",
            default_value=not imagen.safety_checker_enabled,
            callback=lambda tag, value: checkbox_callback(tag, not value),
        )
        dpg.add_text(
            "Check for NSFW image.",
            parent=dpg.add_tooltip("safety_checker"),
        )

        dpg.add_checkbox(
            label="Enable Attention Slicing",
            tag="attention_slicing",
            default_value=imagen.attention_slicing_enabled,
            callback=checkbox_callback,
        )
        dpg.add_text(
            "Slices the computation into multiple steps. Increases performance on MPS.",
            parent=dpg.add_tooltip("attention_slicing"),
        )

        dpg.add_checkbox(
            label="Enable Vae Slicing",
            tag="vae_slicing",
            default_value=imagen.vae_slicing_enabled,
            callback=checkbox_callback,
        )
        dpg.add_text(
            "VAE decodes one image at a time.",
            parent=dpg.add_tooltip("vae_slicing"),
        )

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
        dpg.add_text(
            "Offloads the model onto the CPU. Reduces memory usage while keeping performance at best.",
            parent=dpg.add_tooltip("model_cpu_offload"),
        )

        dpg.add_checkbox(
            label="Enable Compel Prompt Weighting",
            tag="compel_weighting",
            default_value=imagen.compel_weighting_enabled,
            callback=checkbox_callback,
        )
        dpg.add_text(
            "Use compel prompt weighting. + to increase weight and - to decrease.",
            parent=dpg.add_tooltip("compel_weighting"),
        )

        dpg.add_checkbox(
            label="Enable LPW Stable Diffusion Pipeline",
            tag="lpwsd_pipeline",
            default_value=imagen.lpw_stable_diffusion_used,
            callback=lpwsd_callback,
        )
        dpg.add_text(
            "Use LPWSD pipeline. Adds prompt weighting as seen in A1111's webui and long prompts.",
            parent=dpg.add_tooltip("lpwsd_pipeline"),
        )

        dpg.add_button(
            label="Save model weights",
            tag="save_model",
            callback=save_model_callback,
        )
        dpg.add_text(
            "Used to convert a ckpt or safetensors file into diffusers format.",
            parent=dpg.add_tooltip("save_model"),
        )

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
            callback=lambda: toggle_item("upscale_window"),
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
