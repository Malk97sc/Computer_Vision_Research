#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv){
    if(argc < 2){
        std::cerr << "Send Image\n";
        return EXIT_FAILURE;
    }
    
    Mat image = imread(argv[1]);

    if (image.empty()) {
        std::cerr << "Fail to load image\n";
        return EXIT_FAILURE;
    }

    imshow("Image", image);
    waitKey(0);

    return EXIT_SUCCESS;
}