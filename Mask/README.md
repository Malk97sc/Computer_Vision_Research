# Mask

This folder contains examples of Image Masking and Segmentation.  
The goal is to isolate regions of interest in grayscale images through thresholding and preprocessing.

## OpenCV Functions

This folder contains OpenCV experiments using the functions for Image Segmentation and Region Mask.

### Segment Region

This script segments the largest region in an image using a combination of:
1. Blurring.  
2. Canny edge detection.  
3. Morphological closing.  
4. Contour detection.  
5. Region selection.  

### File(s)

| File               | Description                                                   |
|--------------------|---------------------------------------------------------------|
| `segmentRegion.py` | Finds and highlights the largest region in an image.          |

### Usage

Run the script:

```bash
python segmentRegion.py
```

## Clean Image

Removes the minimum pixel value from each row of the image to normalize intensity before segmentation.  

### Files

| File             | Description                                             |
|------------------|---------------------------------------------------------|
| `clean_image.py` | Function to clean grayscale images (row-wise normalization). |


## Segment Image

Segments an image by applying a threshold.  
Pixels greater than the threshold are set to white (255), otherwise black (0).

### Files

| File                | Description                                                |
|---------------------|------------------------------------------------------------|
| `segment_image.py`  | Function to segment grayscale images using a threshold.    |


## Test Script

Demonstrates how to apply cleaning and segmentation together on a sample image.  

### File(s)

| File       | Description                                                 |
|------------|-------------------------------------------------------------|
| `test.py`  | Example script combining cleaning and segmentation.         |
| `example.ipynb` | Jupyter notebook with an interactive demonstration.    |

## Usage

Run the test script:

```bash
python test.py
