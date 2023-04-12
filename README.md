# onGAU: Oversimplified Neural Generative AI UI

![interface of onGAU on Mac](https://raw.githubusercontent.com/sertdfyguhi/onGAU/master/interface.png)

A very simple AI image generator UI interface made using [Dear PyGui](https://github.com/hoffstadt/DearPyGui).

## Installation

1. clone this repository

```sh
git clone https://github.com/sertdfyguhi/onGAU.git
```

2. install requirements.txt

```sh
pip3 install -r requirements.txt
```

3. set the model (from huggingface or locally in `onGAU/models` folder in diffusers format), device, and other configuration in `onGAU/config.py`

4. run `onGAU/main.py`

```sh
python3 onGAU/main.py
```

## Todo

- [x] Show total generation time
- [ ] Save and load prompts and config
- [x] Add img2img pipeline to UI
- [ ] Add upscale pipeline to UI
- [ ] Compile code into cross platform executable
- [ ] Optimize code
  - [ ] Use less memory
  - [ ] Use better code to get image size
