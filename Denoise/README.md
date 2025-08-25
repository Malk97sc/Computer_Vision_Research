# Denoise

This folder contains simple implementations of image denoising techniques.  
The goal is to reduce noise in images using classical filters.

## Gaussian Filter

A smoothing filter that reduces noise by averaging pixels with weights given by a Gaussian function.  
It's effective for removing Gaussian noise but may blur edges.

### Files

| File            | Description                                |
|-----------------|--------------------------------------------|
| `gaussian.cpp`  | Implementation of Gaussian filter          |
| `gaussian.jpg`  | Example output after applying the filter.  |

### Compile & Run

```bash
g++ gaussian.cpp -o gaussian `pkg-config --cflags --libs opencv4`
./gaussian ../Data/noisy_image.jpg
```

## Median Filter

A non-linear filter that replaces each pixel with the median of its neighbors.
It's especially effective for salt-and-pepper noise.

### Files

| File            | Description                                |
|-----------------|--------------------------------------------|
| `median.cpp`  | Implementation of Median filter.   |
| `median.jpg`  | Example output after applying the filter.  |

## Compile & Run

```bash
g++ median.cpp -o median `pkg-config --cflags --libs opencv4`
./median ../Data/noisy_image.jpg
```
