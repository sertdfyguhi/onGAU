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
            generators.append(gen)
        else:
            seeds.append(gen.seed())
            generators.append(gen)

    return generators, seeds
