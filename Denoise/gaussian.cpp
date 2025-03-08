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

    Mat blurred;
    GaussianBlur(image, blurred, Size(5, 5), 0);

    imshow("Image", image);
    imshow("Gaussian Blur", blurred);

    imwrite("gaussian.jpg", blurred);

    waitKey(0);
    destroyAllWindows();
    return EXIT_SUCCESS;
}