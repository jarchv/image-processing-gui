#ifndef tools_cuh
#define tools_cuh

__global__ void swapPixels(unsigned char* dev_src, unsigned char* dev_res, int row, int cols);

unsigned char* readBMPFile( char const*  filename,
    int& width,
    int& height,
    int& depth);

#endif