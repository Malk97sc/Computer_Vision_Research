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

    Mat median;
    medianBlur(image, median, 5);

    imshow("Image", image);
    imshow("Gaussian Blur", median);

    imwrite("median.jpg", median);

    waitKey(0);
    destroyAllWindows();
    return EXIT_SUCCESS;
}