from termcolor import colored
from PIL import Image, UnidentifiedImageError
from sys import argv
import os

HELP = f"""

{colored("python3 pnginfo.py [file]", "green")}
shows the png metadata of a .png file"""


def error(msg):
    print(colored("error: " + msg, "red") + HELP)
    exit(1)


if len(argv) == 1:
    error("file not provided")

file = argv[1]

if not os.path.isfile(file):
    error("file does not exist")

try:
    img = Image.open(file)
except UnidentifiedImageError:
    error("file is not an image")

for key in img.info:
    print(f'{colored(key, attrs=["bold"])}: {repr(img.info[key])}')

print(f'{colored("Image size", attrs=["bold"])}: {img.size[0]} x {img.size[1]}')
