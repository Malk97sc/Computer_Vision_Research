#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

const int CHANNELS = 3;

typedef struct{
    int width, height;
    unsigned char *data;
} Image;

typedef struct{
    Image *input;
    Image *output;
    int width, height;
} ThreadData;

Image readImage(const char *file);
Image toGray(Image img);

void *SobelThread(void *arg);
void applySobelFilter(Image *img, Image **edges, int nImages);
int applySobelKernel(Image *img, int limX, int limY, int *sumX, int *sumY, int dx[][3], int dy[][3]);

void writeImage(const char *file, Image img);

int main(int argc, char **argv){
    if(argc < 3 ){
        perror("Send all the images\n");
        return EXIT_FAILURE;
    }

    Image *images, *edges;
    int nImages= argc - 1;

    images = (Image*) malloc(nImages * sizeof(Image));
    if(!images){
        perror("Memory allocation failed");
        exit(1);
    }

    for(int i=0; i < nImages; i++){
        images[i] = readImage(argv[i+1]); //the 2nd argument start
    }

    applySobelFilter(images, &edges, nImages);

    for(int i=0; i < nImages; i++){
        char outputFileName[22];
        sprintf(outputFileName, "image_%d.pgm", i); //change the file name
        writeImage(outputFileName, edges[i]);
    }

    for(int i=0; i < nImages; i++){
        free(images[i].data);
        free(edges[i].data);
    }
    free(images);
    free(edges);
    return EXIT_SUCCESS;
}

void applySobelFilter(Image *img, Image **edges, int nImages){
    *edges = (Image*)malloc(nImages * sizeof(Image)); //to store all sobel images
    if(!(*edges)){
        perror("Memory allocation failed");
        exit(1);
    }
    
    for(int i=0; i < nImages; i++){
        (*edges)[i].width = img[i].width;
        (*edges)[i].height = img[i].height;
        (*edges)[i].data = (unsigned char *)calloc(img[i].width * img[i].height, sizeof(unsigned char));
        if (!(*edges)[i].data) {
            perror("Memory allocation failed for image data");
            exit(1);
        }
    }

    pthread_t threads[nImages];
    ThreadData threadData[nImages];
    
    for(int i=0; i < nImages; i++){
        threadData[i].input = &img[i];
        threadData[i].height = img[i].height;
        threadData[i].width = img[i].width;
        threadData[i].output = &(*edges)[i];        
        pthread_create(&threads[i], NULL, SobelThread, (void*)&threadData[i]);
    }

    for(int i=0; i < nImages; i++){
        pthread_join(threads[i], NULL); //wait all the threads;
    }
}

void *SobelThread(void *arg){
    ThreadData *data = (ThreadData *)arg;
    Image *img = data->input;
    Image *edge = data->output;
    Image gray = toGray(*img);
    
    int sumX, sumY, pixel, magn;
    int Dx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Dy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for(int x=1; x < data->height-1; x++){
        for(int y=1; y < data->width-1; y++){
            sumX = sumY = 0;
            pixel = applySobelKernel(&gray, x, y, &sumX, &sumY, Dx, Dy);
            magn = (int)sqrt(sumX*sumX + sumY*sumY);
            edge->data[x * edge->width + y] = (magn > 255) ? 255 : magn;
        }
    }

    free(gray.data); 
    pthread_exit(NULL);
}

int applySobelKernel(Image *img, int limX, int limY, int *sumX, int *sumY, int dx[][3], int dy[][3]){
    int pixel = 0;
    for(int i=-1; i <= 1; i++){
        for(int j=-1; j <= 1; j++){
            pixel = img->data[(limX+i) * img->width + (limY+j)];
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
    gray.data = (unsigned char*) malloc(img.width * img.height * sizeof(unsigned char));

    if(!gray.data){
        perror("Memory allocation failed");
        exit(1);
    }

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

    if (!img.data) {
        perror("Memory allocation failed");
        fclose(fl);
        exit(1);
    }

    fread(img.data, CHANNELS, widthAux * heightAux, fl);
    fclose(fl);
    return img;
}

void writeImage(const char *file, Image img) {
    FILE *fl = fopen(file, "wb");
    if (!fl) {
        perror("Fail to load image\n");
        exit(1);
    }
    fprintf(fl, "P5\n%d %d\n255\n", img.width, img.height);
    fwrite(img.data, 1, img.width * img.height, fl); //1 is cause the image is in gray scale
    fclose(fl);
}