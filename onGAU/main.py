from callbacks import *
import logger
import config

import dearpygui.dearpygui as dpg
import pyperclip
import atexit

CENTER = (config.WINDOW_SIZE[0] / 2, config.WINDOW_SIZE[1] / 2)


def add_tooltip(text: str):
    dpg.add_text(
        text,
        parent=dpg.add_tooltip(dpg.last_item()),
    )


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
        width=config.ITEM_WIDTH * 0.75,
    )
    dpg.add_checkbox(label="Load Model", tag="load_model_checkbox", default_value=True)

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
    add_tooltip("The amount to upscale the image.")

    dpg.add_button(
        label="Upscale Image",
        tag="upscale_button",
        callback=upscale_image_callback,
    )

with dpg.window(label="Model Merger", show=False, tag="merge_window", pos=CENTER):
    dpg.add_input_text(
        label="Merge Path", width=config.ITEM_WIDTH, tag="merge_path_input"
    )
    dpg.add_input_text(label="Model 1", tag="model1_input", width=config.ITEM_WIDTH)
    dpg.add_input_text(label="Model 2", tag="model2_input", width=config.ITEM_WIDTH)
    dpg.add_input_text(label="Model 3", tag="model3_input", width=config.ITEM_WIDTH)

    dpg.add_combo(
        items=[
            "Weighted Sum",
            "Add Difference",
            "Sigmoid",
            "Inverse Sigmoid",
        ],
        default_value="Weighted Sum",
        label="Interpolation Method",
        tag="interp_method_input",
        width=config.ITEM_WIDTH,
    )
    add_tooltip("The interpolation method to use to merge the models.")

    dpg.add_slider_double(
        label="Alpha",
        max_value=1.00,
        min_value=0.00,
        default_value=0.80,
        format="%.01f",
        tag="alpha",
    )
    add_tooltip("The ratio to merge the models. 0 makes it the base model.")

    dpg.add_checkbox(label="Ignore Text Encoder", default_value=True, tag="ignore_te")

    dpg.add_button(
        label="Merge",
        width=config.ITEM_WIDTH / 2,
        callback=merge_checkpoint_callback,
        tag="merge_button",
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
            callback=toggle_merge_window_callback,
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
            format="%.01f",
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
            items=config.SCHEDULERS,
            default_value=user_settings["scheduler"] or imagen.scheduler.__name__,
            width=config.ITEM_WIDTH,
            tag="scheduler",
        )
        add_tooltip("The sampling method to use.")

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
        add_tooltip("Check for NSFW image.")

        dpg.add_checkbox(
            label="Enable Attention Slicing",
            tag="attention_slicing",
            default_value=imagen.attention_slicing_enabled,
            callback=checkbox_callback,
        )
        add_tooltip(
            "Slices the computation into multiple steps. Increases performance on MPS."
        )

        dpg.add_checkbox(
            label="Enable Vae Slicing",
            tag="vae_slicing",
            default_value=imagen.vae_slicing_enabled,
            callback=checkbox_callback,
        )
        add_tooltip("VAE decodes one image at a time.")

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
