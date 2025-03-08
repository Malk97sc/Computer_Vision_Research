#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr <<"Send image"<< endl;
        return EXIT_FAILURE;
    }

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr <<"Fail to load image"<< endl;
        return EXIT_FAILURE;
    }

    int dx = 1, dy = 1, mask = 3;

    Mat grad_x, grad_y, grad;
    Sobel(image, grad_x, CV_16S, dx, 0, mask);
    Sobel(image, grad_y, CV_16S, 0, dy, mask);

    convertScaleAbs(grad_x, grad_x);
    convertScaleAbs(grad_y, grad_y);

    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad);

    imshow("Original", image);
    imshow("Gradient Magnitude", grad);

    imwrite("Sobel.jpg", grad);

    waitKey(0);
    return EXIT_SUCCESS;
}