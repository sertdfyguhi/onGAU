from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)
from safetensors.torch import load_file as _load_file
import torch
import math
import os


def create_torch_generator(
    seed: int | list[int] | None, device: str, generator_amount: int = 1
):
    if type(seed) == int:
        seed = [seed] * generator_amount

    generators = []
    seeds = []

    for i in range(generator_amount):
        gen = torch.Generator(device=device)

        if seed:
            gen.manual_seed(s := seed[i % len(seed)])
            seeds.append(s)
        else:
            seeds.append(gen.seed())

        generators.append(gen)

    return generators, seeds


class InterpolationFuncs:
    @staticmethod
    def weighted_sum(theta0, theta1, alpha):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    # Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    @staticmethod
    def sigmoid(theta0, theta1, alpha):
        alpha = alpha * alpha * (3 - (2 * alpha))
        return theta0 + ((theta1 - theta0) * alpha)

    # Inverse Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    @staticmethod
    def inv_sigmoid(theta0, theta1, alpha):
        alpha = 0.5 - math.sin(math.asin(1.0 - 2.0 * alpha) / 3.0)
        return theta0 + ((theta1 - theta0) * alpha)

    @staticmethod
    def add_diff(theta0, theta1, theta2, alpha):
        return theta0 + (theta1 - theta2) * (1.0 - alpha)


def _load_attr(pipe, path: str, attr: str, device: str):
    if not path:
        return

    if pipe:
        return getattr(pipe, attr).state_dict()
    else:
        dir_path = os.path.join(path, attr)
        module_path = next(
            (
                os.path.join(dir_path, file)
                for file in os.listdir(dir_path)
                if file.endswith((".safetensors", ".bin"))
            ),
            None,
        )

        if not module_path:
            raise FileNotFoundError(f"{attr} model could not be found.")

        return (
            _load_file(module_path, device=device)
            if module_path.endswith(".safetensors")
            else torch.load(module_path, map_location=device)
        )


IGNORE_ATTRS = [
    "scheduler",
    "tokenizer",
    "safety_checker",
    "feature_extractor",
    "requires_safety_checker",
]


# Modified version of https://github.com/huggingface/diffusers/blob/main/examples/community/checkpoint_merger.py
# This function modifies imagen1 in place.
@torch.no_grad()
def merge(
    alpha: float,
    interp_func,
    imagen,
    path2: str,
    path3: str = None,
    ignore_te: bool = False,
):
    device = imagen.device
    pipe1 = imagen._pipeline
    pipe2 = (
        download_from_original_stable_diffusion_ckpt(
            path2,
            from_safetensors=path2.endswith(".safetensors"),
            load_safety_checker=False,
        ).to(device)
        if path2.endswith((".ckpt", ".safetensors"))
        else None
    )
    pipe3 = (
        download_from_original_stable_diffusion_ckpt(
            path3,
            from_safetensors=path3.endswith(".safetensors"),
            load_safety_checker=False,
        ).to(device)
        if path3 and path3.endswith((".ckpt", ".safetensors"))
        else None
    )

    # Find each module's state dict.
    for attr in pipe1.config.keys():
        if attr in IGNORE_ATTRS or (ignore_te and attr == "text_encoder"):
            continue

        if not attr.startswith("_"):
            # For an attr if both checkpoint_path_1 and 2 are None, ignore.
            # If atleast one is present, deal with it according to interp method, of course only if the state_dict keys match.
            # if checkpoint_path_1 is None and checkpoint_path_2 is None:
            #     print(f"Skipping {attr}: not present in 2nd or 3d model")
            #     continue

            try:
                module = getattr(pipe1, attr)

                # ignore requires_safety_checker boolean
                if type(module) == bool or module is None:
                    continue

                theta_0 = module.state_dict()
                theta_1 = _load_attr(pipe2, path2, attr, device)
                theta_2 = _load_attr(pipe3, path3, attr, device)

                if theta_0.keys() != theta_1.keys():
                    print(f"Skipping {attr}: key mismatch")
                    continue
                elif theta_2 and theta_1.keys() != theta_2.keys():
                    print(f"Skipping {attr}: y mismatch")
                    continue
            except Exception as e:
                print(f"Skipping {attr} due to an unexpected error: {str(e)}")
                continue

            print(f"Merging {attr}...")

            for key in theta_0.keys():
                # if theta_2:
                #     theta_0[key] = theta_func(
                #         theta_0[key], theta_1[key], theta_2[key], alpha
                #     )
                # else:
                theta_0[key] = (
                    interp_func(theta_0[key], theta_1[key], theta_2[key], alpha)
                    if interp_func == InterpolationFuncs.add_diff
                    else interp_func(theta_0[key], theta_1[key], alpha)
                )

            del theta_1
            del theta_2

            module.load_state_dict(theta_0)
            # module.to(device)

            del theta_0


def progress_callback(step: int, step_count: int, elapsed_time: float, latents):
    # Calculate the percentage
    progress = step / step_count
    eta = (step_count - step) * elapsed_time

    print(
        f"{round(progress * 100)}% {elapsed_time:.1f}s {step}/{step_count} ETA: {eta:.1f}s"
    )
