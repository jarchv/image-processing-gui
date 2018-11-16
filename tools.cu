#include <stdio.h>
#include <stdlib.h>
#include "tools.cuh"

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
    
    printf("\nSize of InfoHeader: %d\n", sizeIH);

    unsigned char infoHeader[sizeIH - 4];

    fread(infoHeader, sizeof(unsigned char), sizeIH - 4 , f);
    
    width           = *(int *)&infoHeader[0];
    height          = *(int *)&infoHeader[4];
    depth           = *(int *)&infoHeader[10]; 
    
    int imageSize   = *(int *)&infoHeader[16];

    printf("\nwidth = %d, height = %d, depth = %d, imageSize = %d\n",
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
        printf("\nSize of ColorTable : %d\n", sizeCT);

        //float pad  = (float)depth / 8;
        //sizeInput  = (int)(width * height * pad);
        
        unsigned char* dataEnc = new unsigned char[imageSize];       
        
        fread(dataEnc, sizeof(unsigned char), imageSize, f);
        fclose(f);    
        

        unsigned char iClr;
        /*
        unsigned char indColor0;
        unsigned char indColor1;
        unsigned char indColor2;
        unsigned char indColor3;
        unsigned char indColor4;
        unsigned char indColor5;
        unsigned char indColor6;
        unsigned char indColor7;
        */
        for (int i = 0; i < imageSize; i++)
        {
            if (depth == 1){

                for (int bit = 0; bit < 8; bit++)
                {
                    iClr = (dataEnc[i] >> bit) & 1;

                    data[i*24 + 3*(8 - bit - 1)    ]  = ColorTable[iClr*4+1];
                    data[i*24 + 3*(8 - bit - 1) + 1]  = ColorTable[iClr*4+1];
                    data[i*24 + 3*(8 - bit - 1) + 2]  = ColorTable[iClr*4+1];
                }
                /*
                unsigned char  X = dataEnc[i];
                indColor7 = (X >> 0) & 1;
                indColor6 = (X >> 1) & 1;
                indColor5 = (X >> 2) & 1;
                indColor4 = (X >> 3) & 1;
                indColor3 = (X >> 4) & 1;
                indColor2 = (X >> 5) & 1;
                indColor1 = (X >> 6) & 1;
                indColor0 = (X >> 7) & 1;
                
                
                printf("ind0 = %2d, ind1 = %2d, ind2 = %2d, ind3 = %2d, ind4 = %2d, ind5 = %2d\n", 
                        indColor0,
                        indColor1,
                        indColor2,
                        indColor3,        
                        indColor4, 
                        indColor5);

                data[i*24    ]  = ColorTable[indColor0*4+1];
                data[i*24 + 1]  = ColorTable[indColor0*4+1];
                data[i*24 + 2]  = ColorTable[indColor0*4+1];

                data[i*24 + 3]  = ColorTable[indColor1*4+1];
                data[i*24 + 4]  = ColorTable[indColor1*4+1];
                data[i*24 + 5]  = ColorTable[indColor1*4+1];

                data[i*24 + 6]  = ColorTable[indColor2*4+1];
                data[i*24 + 7]  = ColorTable[indColor2*4+1];
                data[i*24 + 8]  = ColorTable[indColor2*4+1];

                data[i*24 + 9 ] = ColorTable[indColor3*4+1];
                data[i*24 + 10] = ColorTable[indColor3*4+1];
                data[i*24 + 11] = ColorTable[indColor3*4+1];

                data[i*24 + 12] = ColorTable[indColor4*4+1];
                data[i*24 + 13] = ColorTable[indColor4*4+1];
                data[i*24 + 14] = ColorTable[indColor4*4+1];

                data[i*24 + 15] = ColorTable[indColor5*4+1];
                data[i*24 + 16] = ColorTable[indColor5*4+1];
                data[i*24 + 17] = ColorTable[indColor5*4+1];

                data[i*24 + 18] = ColorTable[indColor6*4+1];
                data[i*24 + 19] = ColorTable[indColor6*4+1];
                data[i*24 + 20] = ColorTable[indColor6*4+1];

                data[i*24 + 21] = ColorTable[indColor7*4+1];
                data[i*24 + 22] = ColorTable[indColor7*4+1];
                data[i*24 + 23] = ColorTable[indColor7*4+1];
                */
            }

            else if (depth == 4){
                for (int bit = 0; bit < 8; bit+=4)
                {
                    iClr = (dataEnc[i] >> bit) & 15;

                    data[i*6 + 3*(1 - bit/4)    ]  = ColorTable[iClr*4+0];
                    data[i*6 + 3*(1 - bit/4) + 1]  = ColorTable[iClr*4+1];
                    data[i*6 + 3*(1 - bit/4) + 2]  = ColorTable[iClr*4+2];
                }

                /*
                indColor1 = 15 &  dataEnc[i];
                indColor0 = 15 & (dataEnc[i] >> 4);
         
                //printf("ind1 = %2d, ind2 = %2d\n", indColor1, indColor2);

                data[i*6    ] = ColorTable[indColor0*4  ];
                data[i*6 + 1] = ColorTable[indColor0*4+1];
                data[i*6 + 2] = ColorTable[indColor0*4+2];

                data[i*6 + 3] = ColorTable[indColor1*4  ];
                data[i*6 + 4] = ColorTable[indColor1*4+1];
                data[i*6 + 5] = ColorTable[indColor1*4+2];

                */
            }

            else if (depth == 8){
                iClr = dataEnc[i];

                data[i*3    ] = ColorTable[iClr*4];
                data[i*3 + 1] = ColorTable[iClr*4+1];
                data[i*3 + 2] = ColorTable[iClr*4+2];

            }


        }
    } else {   

        float pad  = (float)depth / 8;
        sizeInput  = (int)(width * height * pad);     
        printf("depth1 = %d, width = %d, height = %d, sizeInput = %d\n", depth, width, height, sizeInput);
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
