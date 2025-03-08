#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Send image" << endl;
        return EXIT_FAILURE;
    }

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Fail to load image" << endl;
        return EXIT_FAILURE;
    }

    Mat blurred;
    GaussianBlur(image, blurred, Size(5, 5), 0);

    int kernelSize = 3;
    int lowerThreshold = 50; // Lower threshold for Canny
    int upperThreshold = 150; // Upper threshold for Canny
    Mat edges;
    Canny(blurred, edges, lowerThreshold, upperThreshold, kernelSize);

    imshow("Original Image", image);
    imshow("Canny Edges", edges);

    imwrite("canny.jpg", edges);

    waitKey(0);
    destroyAllWindows();

    return EXIT_SUCCESS;
}