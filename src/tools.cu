#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "cudaHeaders.h"
#include "tools.h"

__global__ void swapPixels(unsigned char* dev_src, unsigned char* dev_res, int rows, int cols)
{
    int x       = threadIdx.x + blockIdx.x * blockDim.x;
    int y       = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((x < rows) & (y < cols))
        dev_res[x * cols + y] = dev_src[(rows - x - 1) * cols + y];
}

__global__ void meanFilter_gpu(unsigned char* dev_src, unsigned char* dev_res, int rows, int cols, float* dev_kernel)
{
    int x       = threadIdx.x + blockIdx.x * blockDim.x;
    int y       = threadIdx.y + blockIdx.y * blockDim.y;

    int lim_inf = 0;
    long lim_sup = rows * cols;

    if ((x < rows) & (y < cols))
    {
        long pos = x * cols + y;
        float tmp = (float)dev_src[pos]*dev_kernel[4];

        tmp += ((float)dev_src[pos -   3]*dev_kernel[3]);
        tmp += ((float)dev_src[pos +   3]*dev_kernel[5]);

        tmp += ((float)dev_src[pos - cols - cols/3]*dev_kernel[1]);
        tmp += ((float)dev_src[pos + cols + cols/3]*dev_kernel[7]);

        tmp += ((float)dev_src[pos - cols - cols/3 - 3]*dev_kernel[0]);
        tmp += ((float)dev_src[pos - cols - cols/3 + 3]*dev_kernel[2]);

        tmp += ((float)dev_src[pos + cols + cols/3 - 3]*dev_kernel[6]);
        tmp += ((float)dev_src[pos + cols + cols/3 + 3]*dev_kernel[8]);


        /*
        * Cruz
        * ========
        */

        /*
        if (pos - 3     > lim_inf)
            tmp += ((float)dev_src[pos -    3]*dev_kernel[0]);
        else
            tmp += (float)dev_src[pos];
        if (pos + 3     < lim_sup)
            tmp += ((float)dev_src[pos +   3]*dev_kernel[0]);
        else
            tmp += (float)dev_src[pos];

        if (pos - cols - cols/3  > lim_inf)
            tmp += ((float)dev_src[pos - cols - cols/3]*dev_kernel[0]);
        else
            tmp += (float)dev_src[pos];

        if (pos + cols + cols/3  > lim_inf)
            tmp += ((float)dev_src[pos + cols + cols/3]*dev_kernel[0]);
        else
            tmp += (float)dev_src[pos];

        */
        /*
        * Esquinas
        * ========
        */
        /*
        if (pos - cols - cols/3 - 3 > lim_inf)
            tmp += ((float)dev_src[pos - cols - cols/3 - 3]*dev_kernel[0]);
        else
            tmp += (float)dev_src[pos];

        if (pos - cols - cols/3 + 3 > lim_inf)
            tmp += ((float)dev_src[pos - cols - cols/3 + 3]*dev_kernel[0]);
        else
            tmp += (float)dev_src[pos];

        if (pos + cols + cols/3 - 3 > lim_inf)
            tmp += ((float)dev_src[pos + cols + cols/3 - 3]*dev_kernel[0]);
        else
            tmp += (float)dev_src[pos];           
        if (pos + cols + cols/3 + 3 > lim_inf)
            tmp += ((float)dev_src[pos + cols + cols/3 + 3]*dev_kernel[0]);
        else
            tmp += (float)dev_src[pos];   
        */
        dev_res[pos] = (unsigned char)tmp; 
    }
}
unsigned char * FilterOp(unsigned char* data,  int height, int width, float* kernel)
{
    
    unsigned char *dev_data;
    unsigned char *dev_res;
    float         *dev_kernel;

    int sizeImg = width * height * 3;

    unsigned char* res  = new unsigned char[sizeImg];
    cudaMalloc((void**)&dev_data, sizeImg * sizeof(unsigned char));
    cudaMalloc((void**)&dev_res , sizeImg * sizeof(unsigned char));
    cudaMalloc((void**)&dev_kernel , sizeImg * sizeof(unsigned char));

    cudaMemcpy(dev_data, data, sizeImg * sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_res , res , sizeImg * sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernel , kernel , sizeImg * sizeof(unsigned char),cudaMemcpyHostToDevice);

    int DIM1 = height;
    int DIM2 = width * 3;

    dim3 grids(DIM1/16 + 1, DIM2/16 + 1);
    dim3 threads(16,16);

    meanFilter_gpu<<<grids, threads>>>(dev_data, dev_res, DIM1, DIM2, dev_kernel);

    cudaMemcpy(data, dev_data, sizeImg * sizeof(unsigned char),cudaMemcpyDeviceToHost);
    cudaMemcpy(res , dev_res , sizeImg * sizeof(unsigned char),cudaMemcpyDeviceToHost);

    cudaFree(dev_data);
    cudaFree(dev_res);
    cudaFree(dev_kernel);

    return res;

}
unsigned char *meanFilter(unsigned char* data, int height, int width)
{
    //int sizeOutput = width * height *3;
    
    //unsigned char* res  = new unsigned char[sizeOutput];

    unsigned char* res;

    float *kernel = new float[9];


    kernel[0] = 1.0/9.0;
    kernel[1] = 1.0/9.0;
    kernel[2] = 1.0/9.0;
    kernel[3] = 1.0/9.0;
    kernel[4] = 1.0/9.0;
    kernel[5] = 1.0/9.0;
    kernel[6] = 1.0/9.0;
    kernel[7] = 1.0/9.0;
    kernel[8] = 1.0/9.0;

    res = FilterOp(data , height, width, kernel);
    /*
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

    meanFilter_gpu<<<grids, threads>>>(dev_data, dev_res, DIM1, DIM2);

    cudaMemcpy(data, dev_data, sizeOutput * sizeof(unsigned char),cudaMemcpyDeviceToHost);
    cudaMemcpy(res , dev_res , sizeOutput * sizeof(unsigned char),cudaMemcpyDeviceToHost);

    cudaFree(dev_data);
    cudaFree(dev_res);
    */
    return res;
}


unsigned char* readBMPFile( char const*  filename,
    int& width,
    int& height)
{
    int depth;
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

    /*
    * Monochromatic
    * =============
    if (imageSize != filesize)
        std::cout << "Not equal imageSize: " << imageSize << std::endl;
        imageSize = filesize;
    */
    printf("width = %d, height = %d, depth = %d, imageSize = %d\n",
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

        free(dataEnc);
        fclose(f); 
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

    free(data);
    return res; 
    
}