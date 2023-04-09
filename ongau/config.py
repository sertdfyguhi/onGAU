# User config
SAVE_FILE_PATTERN = (
    "saves/saved%s.png"  # how the program saves the generated images (relative to CWD)
)
DEFAULT_MODEL = "models/anything-v3.0"  # the model to use (either a local path (in diffusers format)) or a huggingface model (for example: stabilityai/stable-diffusion))
DEVICE = "mps"  # device to use. nvidia gpu should use "cuda" and cpu should use "cpu". more info here: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
DEFAULT_IMAGE_SIZE = (512, 512)  # the default image size to use (width, height)

# UI config
FONT = "DankMono-Regular.otf"  # place custom fonts in fonts directory
FONT_SIZE = 17
ITEM_WIDTH = 280
WINDOW_TITLE = "onGAU"
WINDOW_SIZE = (1280, 720)  # width, height
