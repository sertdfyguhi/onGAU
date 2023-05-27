# onGAU: Oversimplified Neural Generative AI UI

![interface of onGAU on Mac](https://raw.githubusercontent.com/sertdfyguhi/onGAU/master/interface.png)

A very simple AI image generator UI interface built with [Dear PyGui](https://github.com/hoffstadt/DearPyGui) and [Diffusers](https://github.com/huggingface/diffusers).

## Installation

1. clone this repository

```sh
git clone https://github.com/sertdfyguhi/onGAU.git
```

2. run `install.sh` to install all necessary modules

```sh
./install.sh
```

3. run `onGAU/main.py` to start the UI

```sh
python3 onGAU/main.py
```

## Scripts

`pnginfo.py`:  
&nbsp;&nbsp;&nbsp;Shows the png metadata (prompt, negative prompt...) of a png file.

`esrgan.py`:  
&nbsp;&nbsp;&nbsp;CLI to upscale an image using ESRGAN.

## Todo

- [x] Show total generation time
- [x] Save and load prompts and config
- [x] Add img2img pipeline
- [x] Add LPW stable diffusion pipeline
- [x] Add textual inversion model loading
- [x] Add Clip Skip parameter
- [x] Add model CPU offloading
- [x] Average step time
- [ ] Generation Progress ETA
- [x] Load .ckpt in app
- [x] Load .safetensors lora in app
- [ ] Implement lora correctly and fully
- [x] Add tooltips
- [x] Denoising strength
- [ ] Change embedding models and loras in app
- [ ] Rework and organize UI
- [ ] Merging models
- [ ] Add controlnet support
- [ ] Add super resolution (ESRGAN/SwinIR) support
  - [x] Add ESRGAN support
  - [ ] Add SwinIR support
- [x] Create an install script to easily install UI
- [x] Interrupt generation process
- [ ] Compile code into cross platform executable
  - [ ] Windows binary
  - [ ] MacOS binary
  - [ ] Linux binary
- Code Optimization
  - [x] Fix memory leak when Compel prompt weighting is enabled
  - [x] Use better code to get image size
  - [x] Sometimes after changing models inference is extremely slow
  - [x] When switching pipelines inference gets extremely slow
  - [ ] First step takes extra time
