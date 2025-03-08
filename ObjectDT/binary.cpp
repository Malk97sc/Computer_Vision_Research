#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv){
    if(argc < 2){
        cerr <<"Send image"<< endl;
        return EXIT_FAILURE;
    }

    Mat image = imread(argv[1]);
    if(image.empty()){
        cerr <<"Fail to load image"<< endl;
        return EXIT_FAILURE;
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    Mat binary;
    threshold(gray, binary, 127, 255, THRESH_BINARY);

    imshow("Original Image", image);
    imshow("Binary Image", binary);
    imwrite("binary.jpg", binary);

    waitKey(0);
    destroyAllWindows();
    return EXIT_SUCCESS;
}