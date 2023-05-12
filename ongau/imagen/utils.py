from safetensors.torch import load_file as _load_file
import torch


def create_torch_generator(
    seed: int | list[int] | None, device: str, generator_amount: int = 1
):
    if type(seed) == int:
        return (
            [torch.Generator(device=device).manual_seed(seed)] * generator_amount,
            [seed] * generator_amount,
        )

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


# https://github.com/huggingface/diffusers/blob/main/scripts/convert_lora_safetensor_to_diffusers.py
# This implementation does break sometimes.
def load_lora(
    pipeline,
    lora_path: str,
    device: str,
    alpha: float = 0.75,
    LORA_PREFIX_TEXT_ENCODER: str = "lora_te",
    LORA_PREFIX_UNET: str = "lora_unet",
):
    # load LoRA weight from .safetensors
    state_dict = _load_file(lora_path, device)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = (
                key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            )
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
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
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = (
                state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            )
            curr_layer.weight.data += alpha * torch.mm(
                weight_up, weight_down
            ).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline.to(device)
