import numpy as np
from Padding import padding

class EdgeDetector:
    def __init__(self, img):
        self.img = img.astype(np.float32)
    
    def _convolve(self, img, kernel):
        if len(img.shape) == 2:
            return self._convolve2gray(img, kernel)
        elif len(img.shape) == 3:
            return self._convolve2rgb(img, kernel)
        else:
            raise ValueError("Unsupported image shape")

    def _convolve2gray(self, img, kernel):
        height, width = img.shape
        kernel_h, kernel_w = kernel.shape
        padded = padding(img, kernel)
        out = np.zeros((height, width), dtype = np.float32)

        for i in range(height):
            for j in range(width):
                reg = padded[i: i+kernel_h, j: j+kernel_w]
                out[i, j] = np.sum(reg * kernel)

        return out

    def _convolve2rgb(self, img, kernel):
        height, width, channels = img.shape
        kernel_h, kernel_w = kernel.shape
        padded = padding(img, kernel)
        out = np.zeros((height, width, channels), dtype=np.float32)

        for c in range(channels):
            for i in range(height):
                for j in range(width):
                    reg = padded[i:i+kernel_h, j:j+kernel_w, c]
                    out[i, j, c] = np.sum(reg * kernel)
        return out
    
    def sobel(self):
        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)

        Gx = self._convolve(self.img, Kx)
        Gy = self._convolve(self.img, Ky)

        magnitude = np.sqrt(Gx**2 + Gy**2)
        magnitude = (magnitude / magnitude.max()) * 255
        return magnitude.astype(np.uint8)

    def laplacian(self):
        K = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]], dtype=np.float32)

        lap = self._convolve(self.img, K)
        lap = np.clip(lap, 0, 255)
        return lap.astype(np.uint8)
