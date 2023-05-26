import sys

sys.path.append("../onGAU")

from argparse import ArgumentParser
from onGAU.imagen import ESRGAN
from PIL import Image
import time

parser = ArgumentParser()
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--image", required=True, type=str)
parser.add_argument("--outpath", required=True, type=str)
parser.add_argument("--device", default="auto", type=str)
parser.add_argument("--upscale", default=None, type=int)
parser.add_argument("--tile_size", default=None, type=int)

args = parser.parse_args()

esrgan = ESRGAN(args.model, args.device)
if args.tile_size:
    esrgan.set_tile_size(args.tile_size)

print("Upscaling image...")

start = time.time()
image = esrgan.upscale_image(Image.open(args.image), args.upscale)

print("Saving image...")

image.image.save(args.outpath)

print(f"Successfully saved image to {args.outpath}.")
print(f"Done in {time.time() - start:.1f}s.")
