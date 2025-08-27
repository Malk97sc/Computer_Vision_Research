# Object Detection (Thresholding & Contours)

This folder contains implementations of object detection using binary thresholding and contour detection. 

## Binary Image

Converts an image into a binary format (black & white) using thresholding.  
This step is fundamental for later contour detection.

### Files

| File          | Description                                    |
|---------------|------------------------------------------------|
| `binary.cpp`  | C++ use of binary thresholding.     |
| `binary.py`   | Python use of binary thresholding.  |
| `binary.jpg`  | Example output of binary thresholding.         |

## Contours

Finds and draws contours of objects from a binary image.  
Contours are useful for detecting object boundaries, counting objects, and shape analysis.

### File(s)

| File            | Description                                         |
|-----------------|-----------------------------------------------------|
| `contours.cpp`  | C++ use of contour detection.            |
| `contours.py`   | Python use of contour detection.         |
| `countour.jpg`  | Example output showing detected contours.           |

---

## Usage

### Python

```bash
python binary.py
python contours.py
```

### C++

```bash
g++ binary.cpp -o binary `pkg-config --cflags --libs opencv4`
g++ contours.cpp -o contours `pkg-config --cflags --libs opencv4`
./binary ../Data/image.jpg
./contours ../Data/image.jpg
```

