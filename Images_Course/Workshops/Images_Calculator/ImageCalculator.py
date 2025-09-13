import numpy as np

class ImageCalculator:
    def __init__(self, img):
        self.img = img.astype(np.uint8)

    #utils
    def img2result(self):
        height, width, channels = self.img.shape
        out = np.zeros((height, width, channels), dtype=np.uint8)
        return height, width, channels, out
    
    #Arithmetic Transformations
    def sum_images(self, imgB, alpha = 0.5):
        beta = 1 - alpha 
        h, w, c = self.img.shape
        h2, w2, _ = imgB.shape

        if h != h2 or w != w2:
            raise ValueError("Both images must have the same size")
        
        out = np.zeros((h, w, c), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                out[i, j] = np.clip(alpha * self.img[i, j] + beta * imgB[i, j], 0, 255)
        return out

    def subtract_images(self, imgB):
        h, w, c = self.img.shape
        h2, w2, _ = imgB.shape

        if h != h2 or w != w2:
            raise ValueError("Both images must have the same size")
        
        out = np.zeros((h, w, c), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                out[i, j] = self.img[i, j] - imgB[i, j]
        return out

    def add_value(self, value = 100):
        h, w, _, out = self.img2result()
        for i in range(h):
            for j in range(w):
                out[i, j] = np.clip(self.img[i, j] + value, 0, 255)
        return out

    def subtract_value(self, value = 100):
        h, w, _, out = self.img2result()
        for i in range(h):
            for j in range(w):
                out[i, j] = np.clip(self.img[i, j] - value, 0, 255)
        return out

    def negative(self, white = 255):
        h, w, _, out = self.img2result()
        for i in range(h):
            for j in range(w):
                out[i, j] = np.clip(abs(white - self.img[i, j]), 0, 255)
        return out

    def square(self):
        h, w, _, out = self.img2result()
        img_float = self.img.astype(np.float32)
        for i in range(h):
            for j in range(w):
                out[i, j] = np.clip(img_float[i, j]**2 / 255, 0, 255).astype(np.uint8)
        return out

    def square_root(self):
        h, w, _, out = self.img2result()
        img_float = self.img.astype(np.float32)
        for i in range(h):
            for j in range(w):
                out[i, j] = np.clip(np.sqrt(img_float[i, j]) * np.sqrt(255), 0, 255).astype(np.uint8)
        return out

    def cube_root(self):
        h, w, _, out= self.img2result()
        img_float = self.img.astype(np.float32)
        for i in range(h):
            for j in range(w):
                out[i, j] = np.clip((img_float[i, j] ** (1/3)) * (255 ** (2/3)), 0, 255).astype(np.uint8)
        return out

    def n_root(self, n = 4):
        if n <= 0:
            raise ValueError("n must be > 0")
        h, w, _, out = self.img2result()
        img_float = self.img.astype(np.float32)
        for i in range(h):
            for j in range(w):
                out[i, j] = np.clip((img_float[i, j] ** (1/n)) * (255 ** (2/n)), 0, 255).astype(np.uint8)
        return out

    #Geometric Transformations
    def translate(self, dx = 10, dy = 10):
        h, w, _, out = self.img2result()
        for i in range(h):
            for j in range(w):
                ni, nj = i + dy, j + dx
                if 0 <= ni < h and 0 <= nj < w:
                    out[ni, nj] = self.img[i, j]
        return out

    def rotate(self, degrees = 90):
        angle = np.radians(degrees)
        h, w, _, out = self.img2result()

        cx, cy = w // 2, h // 2
        for i in range(h):
            for j in range(w):
                x = j - cx
                y = i - cy

                xr = int(x * np.cos(angle) - y * np.sin(angle))
                yr = int(x * np.sin(angle) + y * np.cos(angle))

                xi, yj = xr + cx, yr + cy

                if 0 <= yj < h and 0 <= xi < w:
                    out[i, j] = self.img[yj, xi]
        return out

    def reflection_x(self):
        h, w, _, out = self.img2result()
        for i in range(h):
            for j in range(w):
                out[h - i - 1, j] = self.img[i, j]
        return out

    def reflection_y(self):
        h, w, _, out = self.img2result()
        for i in range(h):
            for j in range(w):
                out[i, w - j - 1] = self.img[i, j]
        return out
