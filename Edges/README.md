# Edges

This folder contains implementations of Edge Detection Algorithms.  
These techniques highlight sharp intensity changes in images, which correspond to edges.

## Canny Algorithms

A algorithm that uses gradient calculation, non-maximum suppression, and hysteresis thresholding.  
It's one of the most widely used edge detection methods.

### Files

| File        | Description                                   |
|-------------|-----------------------------------------------|
| `canny.cpp` | Implementation of Canny edge detector. |
| `canny.jpg` | Example output after applying the filter.     |

### Compile & Run

```bash
g++ canny.cpp -o canny `pkg-config --cflags --libs opencv4`
./canny ../Data/image.jpg
```

## Laplacian

A second-order derivative operator that detects regions of rapid intensity change.
It's sensitive to noise, but highlights edges strongly.

### Files

| File        | Description                                   |
|-------------|-----------------------------------------------|
| `laplacian.cpp` | Implementation of laplacian operator. |
| `laplacian.jpg` | Example output after applying the filter.     |

### Compile & Run

```bash
g++ laplacian.cpp -o laplacian `pkg-config --cflags --libs opencv4`
./laplacian ../Data/image.jpg
```

## Sobel Operator

A gradient-based method that detects horizontal and vertical edges using convolution kernels.
It's less sensitive to noise than the Laplacian.

### Files

| File        | Description                                   |
|-------------|-----------------------------------------------|
| `sobel.cpp` | Implementation of sobel operator. |
| `sobelOpencv.cpp` | Sobel operator using OpenCV built-in function. |
| `sobel.jpg` | Example output after applying the filter.     |

### Compile & Run

```bash
g++ sobel.cpp -o sobel `pkg-config --cflags --libs opencv4`
./sobel ../Data/image.jpg
```
