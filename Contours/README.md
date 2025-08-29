# Contours

This folder contains an implementation of contour detection using the **Moore-Neighbor Tracing Algorithm**. The goal is to trace the boundary of connected white regions (value `255`) in a binary image.

## Moore-Neighbor Tracing Algorithm

The algorithm is a contour-following method that iteratively checks the 8-connected neighborhood of a pixel to trace an object’s boundary.

## Implementation

The core implementation is provided in the `FindContour` class, which includes:

- **`find_start()`**: Finds the first white pixel (255) in the binary image, marking the starting point of the contour.  
- **`search_contour()`**: Applies the Moore-Neighbor tracing algorithm to follow the contour until it returns to the starting point.  
- **`contour2CV()`**: Converts the custom contour representation into an OpenCV-compatible format for visualization.

### Files

| File                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `FindContour.py`      | Python class implementing the Moore-Neighbor tracing algorithm.             |
| `moore-Neighbor.ipynb`| Jupyter notebook with the full implementation of the algorithm step by step.|
| `test.ipynb`          | Example notebook showing how to use `FindContour` on a binary image.        |

