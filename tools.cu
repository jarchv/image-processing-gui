#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cudaHeaders.h"
#include "tools.h"

__global__ void swapPixels(unsigned char* dev_src, unsigned char* dev_res, int rows, int cols)
{
    int x       = threadIdx.x + blockIdx.x * blockDim.x;
    int y       = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < rows and y < cols)
        dev_res[x * cols + y] = dev_src[(rows - x - 1) * cols + y];
}

unsigned char* readBMPFile( char const*  filename,
    int& width,
    int& height,
    int& depth)
{
    FILE* f = fopen(filename, "rb");
    if (f == NULL)
        return NULL;

    /*
    * Header
    * ======
    */
    unsigned char   Header[14];

    fread(Header, sizeof(unsigned char), 14, f);
    
    int filesize = *(int *)&Header[2 ];//Header[5 ] << 24 | Header[4 ] << 16 | Header[3 ] << 8 | Header[2 ];
    int offset   = *(int *)&Header[10];
    
    printf("Filesize = %d, DataOffset = %d\n", filesize, offset);
    
    /*
    * Info Header 
    * ===========
    */

    unsigned char infoHeaderSize[4];

    fread(infoHeaderSize, sizeof(unsigned char), 4, f);
    int sizeIH = *(int *)&infoHeaderSize[0];
    //int sizeIH = infoHeaderSize[3] << 24 | infoHeaderSize[2] << 16 | infoHeaderSize[1] << 8 | infoHeaderSize[0];
    
    printf("Size of InfoHeader: %d\n", sizeIH);

    unsigned char infoHeader[sizeIH - 4];

    fread(infoHeader, sizeof(unsigned char), sizeIH - 4 , f);

    width   = *(int *)&infoHeader[0];
    height  = *(int *)&infoHeader[4];
    depth   = *(int *)&infoHeader[10]; 

    int imageSize   = *(int *)&infoHeader[16];

    printf("width = %d, height = %d, depth = %d\n, imageSize = %d\n",
                                                width, 
                                                height,
                                                depth,
                                                imageSize);  
    
    int sizeInput;
    int sizeOutput = width * height * 3;

    unsigned char* data = new unsigned char[width * height *3];
    unsigned char* res  = new unsigned char[width * height *3];

    if (depth < 24){
        int sizeCT = offset - 14 - sizeIH;
        unsigned char ColorTable[sizeCT];
        fread(ColorTable, sizeof(unsigned char), sizeCT, f);
        printf("Size of ColorTable : %d\n", sizeCT);

        //float pad  = (float)depth / 8;
        //sizeInput  = (int)(width * height * pad);
        
        unsigned char* dataEnc = new unsigned char[imageSize];       
        
        fread(dataEnc, sizeof(unsigned char), imageSize, f);
        fclose(f);    

        int ctColSize   = 4;
        int nPixels     = 8 / depth;
        int nPixelsCh   = nPixels * 3;
        int comptor     = (int)pow(2, depth) - 1; 
        
        unsigned char iClr;

        if (depth == 1)
        {
            for (int i = 0; i < imageSize; i++)
            {
                for (int bit = 0; bit < 8 / depth; bit++)
                {
                    iClr = (dataEnc[i] >> bit * depth) & comptor;
                    for (int j = 0; j < 3; j++)
                        data[i * nPixelsCh + 3*(8 - bit - 1) + j] = ColorTable[iClr * ctColSize + 1];
                }
            }
        }
        else
        {
            for (int i = 0; i < imageSize; i++)
            {
                for (int bit = 0; bit < 8 / depth; bit++)
                {
                    iClr = (dataEnc[i] >> bit * depth) & comptor;
                    for (int j = 0; j < 3; j++)
                        data[i * nPixelsCh + 3*(8/depth - bit - 1) + j] = ColorTable[iClr * ctColSize + j];
                }
            }
        }
    } else {   

        float pad  = (float)depth / 8;
        sizeInput  = (int)(width * height * pad);     
        fread(data, sizeof(unsigned char), sizeInput, f);
        fclose(f);    

    }

    unsigned char *dev_data;
    unsigned char *dev_res;

    cudaMalloc((void**)&dev_data, sizeOutput * sizeof(unsigned char));
    cudaMalloc((void**)&dev_res , sizeOutput * sizeof(unsigned char));

    cudaMemcpy(dev_data, data, sizeOutput * sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_res , res , sizeOutput * sizeof(unsigned char),cudaMemcpyHostToDevice);

    int DIM1 = height;
    int DIM2 = width * 3;

    dim3 grids(DIM1/16 + 1, DIM2/16 + 1);
    dim3 threads(16,16);

    swapPixels<<<grids, threads>>>(dev_data, dev_res, DIM1, DIM2);

    cudaMemcpy(data, dev_data, sizeOutput * sizeof(unsigned char),cudaMemcpyDeviceToHost);
    cudaMemcpy(res , dev_res , sizeOutput * sizeof(unsigned char),cudaMemcpyDeviceToHost);

    cudaFree(dev_data);
    cudaFree(dev_res);

    return res; 
    
}