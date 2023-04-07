from PIL.Image import Image
import numpy as np
import torch

def convert_PIL_to_DPG_image(pil_image: Image):
    # create np array and flatten
    array = np.ravel(np.array(pil_image))
    # convert to float array
    array = array.astype("float32")
    # turn rgba values into floating point numbers
    array = array / 255.0

    return array

def create_torch_generator(seed: int | None, device: str, generator_amount: int):
    if seed: return (
        [torch.Generator(device=device).manual_seed(seed)] * generator_amount,
        [seed] * generator_amount
    )

    generators = []
    seeds = []

    for _ in range(generator_amount):
        gen = torch.Generator(device=device)
        seeds.append(gen.seed())
        generators.append(gen)

    return generators, seeds
