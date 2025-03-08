#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Send image" << endl;
        return EXIT_FAILURE;
    }

    Mat image = imread(argv[1]);
    if (image.empty()) {
        cerr << "Fail to load image" << endl;
        return EXIT_FAILURE;
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    float sobel_x[3][3] = {{-1, 0, 1},
                           {-2, 0, 2},
                           {-1, 0, 1}};
    float sobel_y[3][3] = {{-1, -2, -1},
                           {0, 0, 0},
                           {1, 2, 1}};

    Mat kernel_x(3, 3, CV_32F, sobel_x);
    Mat kernel_y(3, 3, CV_32F, sobel_y);

    Mat grad_x, grad_y, grad;
    filter2D(gray, grad_x, CV_32F, kernel_x);
    filter2D(gray, grad_y, CV_32F, kernel_y);

    magnitude(grad_x, grad_y, grad);
    normalize(grad, grad, 0, 255, NORM_MINMAX, CV_8U);

    Mat abs_grad_x, abs_grad_y;
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    imshow("Original", image);
    /*imshow("Sobel X", abs_grad_x);
    imshow("Sobel Y", abs_grad_y);*/
    imshow("Gradient", grad);

    waitKey(0);
    return EXIT_SUCCESS;
}