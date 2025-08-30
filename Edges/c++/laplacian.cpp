#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv){
    if(argc < 2){
        cerr<<"Send image" << endl;
        return EXIT_FAILURE;
    }

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    if(image.empty()){
        cerr <<"Fail to load image"<< endl;
        return EXIT_FAILURE;
    }

    int kernelSize = 3; //mask for the Laplacian
    Mat laplacian, absLaplacian;
    Laplacian(image, laplacian, CV_16S, kernelSize);

    convertScaleAbs(laplacian, absLaplacian);

    imshow("Original Image", image);
    imshow("Laplacian", absLaplacian);

    imwrite("Laplacian.jpg", absLaplacian);

    waitKey(0);
    destroyAllWindows();

    return EXIT_SUCCESS;
}