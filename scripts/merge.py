# Checkpoint merger. DO NOT USE THIS AS MERGING IS ALREADY SUPPORTED IN THE UI.
# bad code warning

from diffusers import StableDiffusionPipeline, schedulers
from argparse import ArgumentParser

parser = ArgumentParser("Checkpoint Merger Script")
parser.add_argument("models", nargs="+", type=str)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--outpath", type=str, required=True)
parser.add_argument(
    "--alpha", type=float, default=0.5, help="The ratio to merge the checkpoints."
)
parser.add_argument(
    "--interp",
    type=str,
    default=None,
    help="The interpolation method to use. 'sigmoid', 'inv_sigmoid', 'add_diff' and empty for weighted sum.",
)

parser.add_argument("--test", type=bool, default=False)
parser.add_argument("--test_steps", type=int, default=12)
parser.add_argument("--test_scheduler", type=str, default="DPMSolverMultistepScheduler")
# parser.add_argument("--ignore_te", type=bool, default=False)

args = parser.parse_args()

print("Loading model...")

models = args.models
pipeline = StableDiffusionPipeline.from_pretrained(
    models[0], custom_pipeline="checkpoint_merger"
)

# if args.ignore_te:


print("Merging checkpoints...")

new = pipeline.merge(models, alpha=args.alpha, interp=args.interp)

print("Successfully merged checkpoints.")

if args.test:
    new.scheduler = getattr(schedulers, args.test_scheduler).from_config(
        new.scheduler.config
    )
    new.enable_attention_slicing()

    continue_prog = False

    while not continue_prog:
        prompt = input("Prompt: ")

        new(prompt=prompt, num_inference_steps=args.test_steps).images[0].show()

        match input("Save model or retry? (y/n/r) ").lower():
            case "n", "no":
                exit(0)

            case "r", "retry":
                break

            case _:
                continue_prog = True

new.save_pretrained(args.outpath)
