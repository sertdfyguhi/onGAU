# onGAU configuration / constants file.

# The pattern path to the directory to save generated images. %s is where the number will be.
SAVE_FILE_PATTERN = "saves/saved%s.png"

# The default model to load if user settings does not contain one. (huggingface model or local model)
DEFAULT_MODEL = "models/breakdomain_M2150"

# The pytorch device to use. Automatically infers it if it is "auto". More info here: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
DEVICE = "auto"

# Array of paths to embedding / textual inversion models to use.
EMBEDDING_MODELS = [
    "models/embeds/bad_prompt.pt",
    "models/embeds/EasyNegative.pt",
    "models/embeds/badhandv4.pt",
]

# Array of paths to lora files to use. format: [(path, weight)]
LORAS = []

# Path to ESRGAN model.
ESRGAN_MODEL = "models/esrgan/RealESRGAN_x4plus_anime_6B.pth"

# Path to the .ini file to save user settings.
USER_SETTINGS_FILE = "onGAU/user_settings.ini"


# Default values if user settings does not contain it. You do not need to change this.
DEFAULT_PROMPT = ""
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_GUIDANCE_SCALE = 8.0
DEFAULT_STEP_COUNT = 20
DEFAULT_IMAGE_AMOUNT = 1
DEFAULT_SEED = ""
DEFAULT_WIDTH = 512  # image width
DEFAULT_HEIGHT = 512  # image height
DEFAULT_UPSCALE_AMOUNT = 4
DEFAULT_PIPELINE = "Text2Img"
# Use long prompt weighting stable diffusion pipeline by default. info: https://huggingface.co/docs/diffusers/v0.16.0/en/using-diffusers/custom_pipeline_examples#long-prompt-weighting-stable-diffusion
DEFAULT_LPWSD_PIPELINE = True

# A list of schedulers to use. You do not need to change this.
SCHEDULERS = [
    "DDIMInverseScheduler",
    "DDIMScheduler",
    "DDPMScheduler",
    "DEISMultistepScheduler",
    "DPMSolverMultistepScheduler",
    "DPMSolverMultistepScheduler++",
    "DPMSolverMultistepScheduler Karras",
    "DPMSolverMultistepScheduler Karras++",
    "DPMSolverSinglestepScheduler",
    "EulerAncestralDiscreteScheduler",
    "EulerAncestralDiscreteScheduler Karras",
    "EulerDiscreteScheduler",
    "HeunDiscreteScheduler",
    "IPNDMScheduler",
    "KDPM2AncestralDiscreteScheduler",
    "KDPM2DiscreteScheduler",
    "KarrasVeScheduler",
    "LMSDiscreteScheduler",
    "PNDMScheduler",
    "RePaintScheduler",
    "ScoreSdeVeScheduler",
    "ScoreSdeVpScheduler",
    "UnCLIPScheduler",
    "UniPCMultistepScheduler",
    "VQDiffusionScheduler",
]

# UI configurations. You do not need to change this.
FONT = "DankMono-Regular.otf"  # place custom fonts in fonts directory
FONT_SIZE = 15
ITEM_WIDTH = 280
WINDOW_TITLE = "onGAU"
WINDOW_SIZE = (1280, 720)  # width, height