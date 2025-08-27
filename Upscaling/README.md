# Upscaling

This folder contains an example of Image Super-Resolution using OpenCV's `dnn_superres` module.  
The goal is to upscale low-resolution images using a pretrained deep learning model.

## Model

We use the **EDSR (Enhanced Deep Residual Network)** pretrained model for ×3 upscaling.  

**Important:**  
You must create a folder called `Model/` inside this directory and place the downloaded model.  

- The download link is provided in a comment inside `superres.py`:  
  [EDSR TensorFlow Models](https://github.com/Saafke/EDSR_Tensorflow/tree/master/models)

### Files

| File              | Description                                    |
|-------------------|------------------------------------------------|
| `EDSR_x3.pb`      | Pretrained TensorFlow model for ×3 upscaling.  |


## Super-Resolution Script

Loads an input image, applies the EDSR ×3 model, and saves the upscaled version.  

### Files

| File            | Description                                                    |
|-----------------|----------------------------------------------------------------|
| `superres.py`   | Script to load an image, upscale it with EDSR, and save output.|
| `upscaled.png`  | Example result of applying super-resolution.                   |

## Usage

Run the script:

```bash
python superres.py
```
