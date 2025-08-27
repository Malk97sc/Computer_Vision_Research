# Pooling

This folder contains examples of pooling operations implemented manually.  
Pooling is a technique commonly used in Convolutional Neural Networks (CNNs) to reduce spatial dimensions while preserving important features.

## Pooling Function

Custom implementation of max pooling and average pooling.  
Works with RGB images using a sliding kernel and stride.

### Files

| File             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `pooling_fc.py`  | Defines a function for max/avg pooling with configurable kernel and stride. |

## Test Script

Applies the custom pooling function on a sample image (`coins.jpg`) and displays:  
- Original image  
- Max pooled image  
- Average pooled image  

Also prints the shape of the images to show the effect of pooling.

### File(s)

| File       | Description                                      |
|------------|--------------------------------------------------|
| `test.py`  | Demonstrates the pooling function on an example. |

## Usage

Run the test script:

```bash
python test.py
```
