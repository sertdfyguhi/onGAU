from safetensors.torch import load_file as _load_file
from collections import defaultdict
import torch


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


# https://github.com/huggingface/diffusers/issues/3064
# This implementation does break sometimes.
def load_lora(
    pipeline,
    lora_path: str,
    device: str,
    weight: float = 0.75,
    dtype: torch.dtype = torch.float32,
):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = _load_file(lora_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split(".", 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():
        if "text" in layer:
            layer_infos = layer.split(f"{LORA_PREFIX_TEXT_ENCODER}_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(f"{LORA_PREFIX_UNET}_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += f"_{layer_infos.pop(0)}"
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems["lora_up.weight"].to(dtype)
        weight_down = elems["lora_down.weight"].to(dtype)
        alpha = elems["alpha"].item() / weight_up.shape[1] if elems["alpha"] else 1.0
        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += (
                weight
                * alpha
                * torch.mm(
                    weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
                )
                .unsqueeze(2)
                .unsqueeze(3)
            )
        else:
            curr_layer.weight.data += weight * alpha * torch.mm(weight_up, weight_down)

    return pipeline
