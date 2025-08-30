import numpy as np

def padding(img, kernel):
    k_h, k_w = kernel.shape[:2]
    pad_h, pad_w = k_h // 2, k_w // 2

    if len(img.shape) == 2:  # Grayscale
        height, width = img.shape
        out = np.zeros((height + 2*pad_h, width + 2*pad_w), dtype=img.dtype)
        out[pad_h:pad_h+height, pad_w:pad_w+width] = img
    elif len(img.shape) == 3:  # Color
        height, width, channels = img.shape
        out = np.zeros((height + 2*pad_h, width + 2*pad_w, channels), dtype=img.dtype)
        out[pad_h:pad_h+height, pad_w:pad_w+width, :] = img
    else:
        raise ValueError("Unsupported image shape")
    
    return out

