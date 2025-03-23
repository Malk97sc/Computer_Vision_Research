#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int CHANNELS = 3;

typedef struct{
    int width, height;
    unsigned char *data;
} Image;

Image readImage(const char *file);
void damageRegion(Image *img, int x, int y, int w, int h);
void writeImage(const char *file, Image img);

int main(int argc, char **argv){
    if(argc < 2){
        perror("Send image\n");
        return EXIT_FAILURE;
    }
    srand(time(NULL));
    int damage = 120, position = 30;
    Image img;

    img = readImage(argv[1]);
    damageRegion(&img, position, position, damage, damage);
    writeImage("damaged.ppm", img);

    free(img.data);
    return EXIT_SUCCESS;
}

void damageRegion(Image *img, int x, int y, int w, int h){
    if(!img || !img->data) return;
    int idx = 0;

    for(int i = x; i < x+w && i < img->width; i++){
        for(int j= y; j < y+h && j < img->height; j++){
            idx = (j * img->width + i) * CHANNELS; //to get the coordinates in (x, y) (RGB)
            img->data[idx] = rand() % 256; 
            img->data[idx+1] = rand() % 256; 
            img->data[idx+2] = rand() % 256;
        }
    }
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
    if(!fl){
        perror("Fail to save image\n");
        exit(1);
    }
    fprintf(fl, "P6\n%d %d\n255\n", img.width, img.height);
    fwrite(img.data, CHANNELS, img.width * img.height, fl); 
    fclose(fl);
}