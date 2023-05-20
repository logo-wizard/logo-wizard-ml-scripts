# Logo Wizard ML Scripts
This repository contains scripts that were used for data processing, model training, as well as for implementation into a web service.

## Colorization
`colorizer` folder contains scripts for image colorization inference. It uses pretrained `iColoriT` model, you can find it [here](https://github.com/pmh9960/iColoriT/tree/main/iColoriT_demo#pretrained-icolorit). Also there is an example of how to use this model.

## Text eraser
`eraser` folder contains scripts for text erasing. It uses two pretrained model: `DBNet++` for text detection and `LaMa` for object deletion. You can use them separately or in tandem, achieving auto text erasing. Also there is an example of how to use this model.  
  
`DBNet++` model is loaded automatically using [mmocr](https://github.com/open-mmlab/mmocr) framework.  
  
#### mmocr installation
```console
pip install -r requirements.txt
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
mim install 'mmdet>=3.0.0rc0'

git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
pip install -e .
```
`LaMa` model weights can be found [here](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt).

## Logo stylization
`styler` folder contains scripts for applying different styles to an image. It uses `ControlNet` with the Canny algorithm for edge detection.  
  
`ControlNet` with `Stable Diffusion` model is loaded automatically using [diffusers](https://github.com/huggingface/diffusers) framework.  

## Data processing
`data_processing` folder contains scripts for captioning collected dataset and for processing survey results. You can find more information in the corresponding notebooks.

## LoRa fine-tuning
`lora` folder contains sripts for Stable Diffusion fine-tuning using LoRa method. It contains two different implementations with LoRa and LoHa (LoRA with Hadamard Product representation). You can find more information in the corresponding notebooks.
