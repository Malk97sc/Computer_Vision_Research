# Image Processing in C/C++/Python

This project demonstrates basic image processing techniques using OpenCV in C++. It includes edge detection, noise reduction, and contour detection.

Use the following command to compile the code:

```bash
g++ canny.cpp -o cany `pkg-config --cflags --libs opencv4`
```
Run:

```bash
./cany ../Data/image.jpg
```

## Non-OpenCv images

To use this content, you need to convert the image to the PPM format. You can use the following link to do this: https://convertio.co/es/jpg-ppm/

To use the program (Edges folder), you can compile it with this command:

```bash
gcc sobelwithThreads.c -lm -o sobelThreads.out
```

To run:

```bash
./sobelThreads.out 2 ../Eye.ppm
```
Note: The first argument is the number of threads, and the second is the image in PPM format.

Note: This has been tested on Linux.
