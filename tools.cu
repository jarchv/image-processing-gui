#include <stdio.h>
#include <stdlib.h>
#include "tools.cuh"

__global__ void swapPixels(unsigned char* dev_src, unsigned char* dev_res, int rows, int cols)
{
    int x       = threadIdx.x + blockIdx.x * blockDim.x;
    int y       = threadIdx.y + blockIdx.y * blockDim.y;
    
    dev_res[x * cols + y] = dev_src[(rows - x - 1) * cols + y];
}

unsigned char* readBMPFile( char const*  filename,
    int& width,
    int& height,
    int& depth)
{
FILE*           f = fopen(filename, "rb");
unsigned char   info[54];

fread(info, sizeof(unsigned char), 54, f);

width   = *(int *)&info[18];
height  = *(int *)&info[22];
depth   = *(int *)&info[28]; 

printf("width = %d, height = %d, depth = %d\n",
width, 
height,
depth);    

int pad = 3;
if (depth < 24)
pad = 1;

int size   = width * height * pad;

unsigned char* data = new unsigned char[size];
unsigned char* res  = new unsigned char[size];

fread(data, sizeof(unsigned char), size, f);
fclose(f);    

unsigned char *dev_data;
unsigned char *dev_res;

cudaMalloc((void**)&dev_data, size * sizeof(unsigned char));
cudaMalloc((void**)&dev_res , size * sizeof(unsigned char));

cudaMemcpy(dev_data, data, size * sizeof(unsigned char),cudaMemcpyHostToDevice);
cudaMemcpy(dev_res , res , size * sizeof(unsigned char),cudaMemcpyHostToDevice);

int DIM1 = height;
int DIM2 = width * pad;

dim3 grids(DIM1/16, DIM2/16);
dim3 threads(16,16);

swapPixels<<<grids, threads>>>(dev_data, dev_res, DIM1, DIM2);

cudaMemcpy(data, dev_data, size * sizeof(unsigned char),cudaMemcpyDeviceToHost);
cudaMemcpy(res , dev_res , size * sizeof(unsigned char),cudaMemcpyDeviceToHost);

cudaFree(dev_data);
cudaFree(dev_res);

return res;
}