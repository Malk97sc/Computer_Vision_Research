#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int CHANNELS = 3;

typedef struct{
    int width, height;
    unsigned char *data;
} Image;

Image readImage(const char *file);
Image toGray(Image img);
Image sobelFilter(Image img);
int applySobelKernel(Image img, int limX, int limY, int *sumX, int *sumY, int dx[][3], int dy[][3]);
void writeImage(const char *file, Image img);

int main(int argc, char **argv){
    if(argc < 2){
        perror("Send image\n");
        return EXIT_FAILURE;
    }
    Image newImage, gray, sobel;
    newImage = readImage(argv[1]);
    gray = toGray(newImage);
    sobel = sobelFilter(gray);
    writeImage("sobel.pgm", sobel);

    free(newImage.data);
    free(gray.data);
    free(sobel.data);
    return EXIT_SUCCESS;
}

Image sobelFilter(Image img){
    Image edge;
    edge.width = img.width;
    edge.height = img.height;
    edge.data = (unsigned char*) calloc(img.width * img.height, sizeof(unsigned char));

    int sumX = 0, sumY = 0, pixel, magn;
    int Dx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Dy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for(int x=1; x < img.height-1; x++){
        for(int y=1; y < img.width-1; y++){
            sumX = sumY = 0;
            pixel = applySobelKernel(img, x, y, &sumX, &sumY, Dx, Dy);
            magn = (int)sqrt(pow(sumX, 2) + pow(sumY, 2));
            edge.data[x * img.width + y] = (magn > 255) ? 255: magn; //if magnitude > 255 dont pass the limits
        }
    }
    return edge;
}

int applySobelKernel(Image img, int limX, int limY, int *sumX, int *sumY, int dx[][3], int dy[][3]){
    int pixel = 0;
    for(int i=-1; i <= 1; i++){
        for(int j=-1; j <=1; j++){
            pixel = img.data[(limX+i) * img.width + (limY+j)];
            *sumX += pixel * dx[i+1][j+1];
            *sumY += pixel * dy[i+1][j+1];
        }
    }
    return pixel;
}

Image toGray(Image img){
    Image gray;
    gray.width = img.width;
    gray.height = img.height;
    gray.data = (unsigned char*) malloc(img.width * img.height);

    unsigned char r, g, b;

    for(int i=0; i < img.width * img.height; i++){
        r = img.data[i * CHANNELS];
        g = img.data[i * CHANNELS + 1];
        b = img.data[i * CHANNELS + 2];
        gray.data[i] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
    }
    return gray;
}

Image readImage(const char *file){
    FILE *fl = fopen(file, "rb");
    if(!fl){
        perror("Fail to load image\n");
        exit(1);
    }

    char format[3];
    int maxval, widthAux, heightAux;
    fscanf(fl, "%2s\n%d %d\n%d\n", format, &widthAux, &heightAux, &maxval);
    if(format[0] != 'P' || format[1] != '6'){
        fprintf(stderr, "Fail format\n");
        fclose(fl);
        exit(1);
    }
    
    Image img;
    img.width = widthAux;
    img.height = heightAux;
    img.data = (unsigned char *)malloc(CHANNELS * widthAux * heightAux);
    fread(img.data, CHANNELS, widthAux * heightAux, fl);
    fclose(fl);
    return img;
}

void writeImage(const char *file, Image img){
    FILE *fl = fopen(file, "wb");
    if (!fl) {
        perror("Fail to load image\n");
        exit(1);
    }
    fprintf(fl, "P5\n%d %d\n255\n", img.width, img.height);
    fwrite(img.data, 1, img.width * img.height, fl); //1 is cause the image is in gray scale
    fclose(fl);
}