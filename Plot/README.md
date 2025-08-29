# Plot

This module provides utilities for image visualization, offering two main approaches:  
- Visualization using OpenC in an interactive window. Util to see each pixel.  
- Visualization using Matplotlib, useful in notebook environments.  

## Functions

### `plot_cv(img)`
Displays an image in a window using OpenCV.  
Execution pauses until a key is pressed, then the window is closed.  

### `plot_img(img, gray=True)`
Displays an image with Matplotlib.  
- By default, the image is shown in grayscale.  
- If `gray=False`, the image is displayed in color.  

## Files

| File          | Description                                                 |
|---------------|-------------------------------------------------------------|
| `plot_cv.py`  | Function for displaying images using OpenCV.            |
| `plot_img.py` | Function for displaying images using Matplotlib.        |
