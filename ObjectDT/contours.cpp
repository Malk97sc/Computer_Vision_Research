#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

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

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourImage = image.clone();
    drawContours(contourImage, contours, -1, Scalar(0, 255, 0), 2);

    imshow("Original Image", image);
    imshow("Contours", contourImage);

    imwrite("countour.jpg", contourImage); //save image
    
    waitKey(0);
    destroyAllWindows();

    return EXIT_SUCCESS;
}