# User config
SAVE_FILE_PATTERN = "saves/saved%s.png" # how the program saves the generated images (relative to CWD)
DEFAULT_MODEL = "models/anything-v3.0" # the model to use (either a local path (in diffusers format) or a huggingface model)
DEVICE = "mps" # the device to use (https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)

# UI config
FONT = "DankMono-Regular.otf" # place custom fonts in fonts directory
FONT_SIZE = 17
ITEM_WIDTH = 280
WINDOW_TITLE = "onGAU"
WINDOW_SIZE = (1280, 720) # width, height