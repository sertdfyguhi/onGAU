import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../onGAU"))

from argparse import ArgumentParser
from imagen import RealESRGAN
from PIL.PngImagePlugin import PngInfo
from PIL import Image
import time


def dict_to_PngInfo(info: dict):
    pnginfo = PngInfo()

    for key in info:
        pnginfo.add_text(key, info[key])

    return pnginfo


parser = ArgumentParser("ESRGAN Upscale Script")
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--image", required=True, type=str)
parser.add_argument("--outpath", required=True, type=str)
parser.add_argument("--device", default="auto", type=str)
parser.add_argument("--upscale", default=None, type=int)
parser.add_argument("--tile_size", default=None, type=int)

args = parser.parse_args()

esrgan = RealESRGAN(args.model, args.device)
if args.tile_size:
    esrgan.set_tile_size(args.tile_size)

print("Upscaling image...")

start = time.time()
image = Image.open(args.image)
upscaled = esrgan.upscale_image(image, args.upscale)

print("Saving image...")

upscaled.image.save(
    args.outpath, exif=image.getexif(), pnginfo=dict_to_PngInfo(image.info)
)

print(f"Successfully saved image to {args.outpath}.")
print(f"Done in {time.time() - start:.1f}s.")
