#include <stdio.h>

int main(int argc, char *argv[])
{
    if(argc < 2 || argc > 3)
    {
        printf("Requires input of image file location");
    }
    //take input for .jpg file
    char *image_file = argv[1];

    //run CNN model
    //run OOD model 
    //if OOD returns unknown class then
    //run trainer
    return 0;
}