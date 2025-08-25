# CNN Architecture (PyTorch)

This folder contains a simple Convolutional Neural Network (CNN) implemented with PyTorch.  
It's a starting point to understand the components of CNNs such as convolutional layers, activation functions, pooling, and fully connected layers.

##  Simple

The architecture is defined in `SimpleCNN`:

- **Conv Layer 1**: 3 input channels → 16 filters (3x3, stride=1, padding=1)  
- **ReLU Activation**  
- **MaxPooling**: 2x2 (downsamples by factor of 2)  

- **Conv Layer 2**: 16 input channels → 32 filters (3x3, stride=1, padding=1)  
- **ReLU Activation**  
- **MaxPooling**: 2x2  

- **Fully Connected 1**: Flattens to `32 * 56 * 56 → 64`  
- **ReLU Activation**  
- **Fully Connected 2**: `64 → 2` (2 output classes)  


##  File(s)

| File                | Description                              |
|---------------------|------------------------------------------|
| `simple_cnn.py`     | Contains the `SimpleCNN` model definition |
