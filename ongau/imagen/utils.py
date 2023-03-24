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

def create_torch_generator(seed: int | None, device: str):
    generator = torch.Generator(device=device)

    if seed:
        generator = generator.manual_seed(seed)
    else:
        generator.seed()

    return generator, generator.initial_seed()
