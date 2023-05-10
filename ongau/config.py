# User config
SAVE_FILE_PATTERN = (
    "saves/saved%s.png"  # how the program saves the generated images (relative to CWD)
)
DEFAULT_MODEL = "models/anything-v3.0"  # the default model to use (either a local path (in diffusers format)) or a huggingface model (for example: stabilityai/stable-diffusion))
DEVICE = "mps"  # device to use. gpu should use "cuda" and cpu should use "cpu". more info here: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
EMBEDDING_MODELS = []  # an array of paths to embedding models to use
USER_SETTINGS_FILE = "onGAU/user_settings.ini"


# dont need to change anything under here
DEFAULT_PROMPT = ""
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_GUIDANCE_SCALE = 8.0
DEFAULT_STEP_COUNT = 20
DEFAULT_IMAGE_AMOUNT = 1
DEFAULT_SEED = ""
DEFAULT_WIDTH = 512  # image width
DEFAULT_HEIGHT = 512  # image height
DEFAULT_PIPELINE = "Text2Img"
# use long prompt weighting stable diffusion pipeline by default. info: https://huggingface.co/docs/diffusers/v0.15.0/en/using-diffusers/custom_pipeline_examples#long-prompt-weighting-stable-diffusion
DEFAULT_LPWSD_PIPELINE = True

# UI config
FONT = "DankMono-Regular.otf"  # place custom fonts in fonts directory
FONT_SIZE = 17
ITEM_WIDTH = 280
WINDOW_TITLE = "onGAU"
WINDOW_SIZE = (1280, 720)  # width, height
