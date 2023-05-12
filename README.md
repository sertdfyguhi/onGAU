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

## Todo

- [x] Show total generation time
- [ ] Save and load prompts and config
- [x] Add img2img pipeline
- [x] Add LPW stable diffusion pipeline
- [x] Add textual inversion model loading
- [x] Add Clip Skip parameter
- [ ] Add CPU model offloading
- [x] Average step time
- [x] Load .ckpt in app
- [x] Load .safetensors lora in app
- [x] Add tooltips
- [ ] Change embedding models and loras in app
- [ ] Add super resolution (ESRGAN/SwinIR) support
- [x] Create an install script to easily install UI
- [x] Interrupt generation process
- [ ] Compile code into cross platform executable
- Code Optimization
  - [x] Fix memory leak when Compel prompt weighting is enabled
  - [x] Use better code to get image size
  - [x] Sometimes after changing models inference is extremely slow
  - [ ] When switching pipelines inference gets extremely slow
