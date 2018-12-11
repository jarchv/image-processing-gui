#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "cudaHeaders.h"
#include "tools.h"

# define M_PI 3.14159265358979323846 
//float M_PI = 3.1416;

__global__ void FilterOp_gpu(unsigned char* dev_src, unsigned char* dev_res, int rows, int cols, float* dev_kernel)
{
    int x       = threadIdx.x + blockIdx.x * blockDim.x;
    int y       = threadIdx.y + blockIdx.y * blockDim.y;

    int lim_inf = 0;
    long lim_sup = rows * cols;

    if ((x < rows) & (y < cols))
    {
        long pos = x * cols + y;
        float tmp = (float)dev_src[pos]*dev_kernel[4];

        /*
        * Cruz
        * ========================================================== 
        */

        if (pos - 3     > lim_inf)
            tmp += ((float)dev_src[pos -    3]*dev_kernel[3]);
        else 
            tmp += ((float)dev_src[pos]*dev_kernel[3]);
            
        if (pos + 3     < lim_sup)
            tmp += ((float)dev_src[pos +   3]*dev_kernel[5]);
        else
            tmp += ((float)dev_src[pos]*dev_kernel[5]);

        if (pos - cols - cols/3  > lim_inf)
            tmp += ((float)dev_src[pos - cols - cols/3]*dev_kernel[1]);
        else
            tmp += ((float)dev_src[pos]*dev_kernel[1]);

        if (pos + cols + cols/3  < lim_sup)
            tmp += ((float)dev_src[pos + cols + cols/3]*dev_kernel[7]);
        else
            tmp += ((float)dev_src[pos]*dev_kernel[7]);

        
        /*
        * Esquinas
        * ========
        */
        
        if (pos - cols - cols/3 - 3 > lim_inf)
            tmp += ((float)dev_src[pos - cols - cols/3 - 3]*dev_kernel[0]);
        else
            tmp += ((float)dev_src[pos]*dev_kernel[0]);

        if (pos - cols - cols/3 + 3 > lim_inf)
            tmp += ((float)dev_src[pos - cols - cols/3 + 3]*dev_kernel[2]);
        else
            tmp += ((float)dev_src[pos]*dev_kernel[2]);

        if (pos + cols + cols/3 - 3 < lim_sup)
            tmp += ((float)dev_src[pos + cols + cols/3 - 3]*dev_kernel[6]);
        else
            tmp += ((float)dev_src[pos]*dev_kernel[6]);           
        if (pos + cols + cols/3 + 3 < lim_sup)
            tmp += ((float)dev_src[pos + cols + cols/3 + 3]*dev_kernel[8]);
        else
            tmp += ((float)dev_src[pos]*dev_kernel[8]);   
        
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

    FilterOp_gpu<<<grids, threads>>>(dev_data, dev_res, DIM1, DIM2, dev_kernel);

    cudaMemcpy(data, dev_data, sizeImg * sizeof(unsigned char),cudaMemcpyDeviceToHost);
    cudaMemcpy(res , dev_res , sizeImg * sizeof(unsigned char),cudaMemcpyDeviceToHost);

    cudaFree(dev_data);
    cudaFree(dev_res);
    cudaFree(dev_kernel);

    return res;

}

unsigned char *meanFilter(unsigned char* data, int height, int width)
{
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
    return res;
}

unsigned char *laplacianFilter(unsigned char* data, int height, int width)
{
    unsigned char* res;
    float *kernel = new float[9];


    kernel[0] =  0.0;
    kernel[1] = -1.0;
    kernel[2] =  0.0;
    kernel[3] = -1.0;
    kernel[4] =  4.0;
    kernel[5] = -1.0;
    kernel[6] =  0.0;
    kernel[7] = -1.0;
    kernel[8] =  0.0;

    res = FilterOp(data , height, width, kernel);
    return res;
}

__global__ void swapPixels(unsigned char* dev_src, unsigned char* dev_res, int rows, int cols)
{
    int x       = threadIdx.x + blockIdx.x * blockDim.x;
    int y       = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((x < rows) & (y < cols))
        dev_res[x * cols + y] = dev_src[(rows - x - 1) * cols + y];
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


__global__ void getSumRGB_gpu(unsigned char* dev_data, float* dev_res, int rows, int cols)
{
    int x       = threadIdx.x + blockIdx.x * blockDim.x;
    int y       = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x < rows) & (y < cols))
    {
        long pos = x * cols + y;
        if ((pos % 3) == 1)
        {
            dev_res[pos/3] = (float)dev_data[pos - 1] + 
                             (float)dev_data[pos    ] + 
                             (float)dev_data[pos + 1];
        }
    }    
}
float * getSumRGB(unsigned char* data, int width, int height)
{
    int sizeGray = width * height;
    
    float* res = new float[sizeGray];

    unsigned char *dev_data;
    float *dev_res;

    cudaMalloc((void**)&dev_data, 3 * sizeGray * sizeof(unsigned char));
    cudaMalloc((void**)&dev_res , sizeGray * sizeof(float));

    cudaMemcpy(dev_data, data, 3 * sizeGray * sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_res , res , sizeGray * sizeof(float),cudaMemcpyHostToDevice);

    int DIM1 = height;
    int DIM2 = width * 3;

    dim3 grids(DIM1/16 + 1, DIM2/16 + 1);
    dim3 threads(16,16);

    getSumRGB_gpu<<<grids, threads>>>(dev_data, dev_res, DIM1, DIM2);

    cudaMemcpy(data, dev_data, 3 * sizeGray * sizeof(unsigned char),cudaMemcpyDeviceToHost);
    cudaMemcpy(res , dev_res , sizeGray * sizeof(float),cudaMemcpyDeviceToHost);
    
    cudaFree(dev_data);
    cudaFree(dev_res);

    return res; 

}   

unsigned char* toGray(unsigned char* src, int width, int height)
{
    float* sumRGB;
    unsigned char* res  = new unsigned char[width * height];

    sumRGB =  getSumRGB(src, width, height);

    for(int i = 0; i < width * height; i++)
    {
        res[i] = (unsigned char)(sumRGB[i]/3);
    }
    free(sumRGB);

    return res;
}

unsigned char* toChromatic(unsigned char* src, int width, int height)
{
    float* sumRGB;
    unsigned char* res  = new unsigned char[width * height * 3];

    sumRGB =  getSumRGB(src, width, height);

    for(int i = 0; i < width * height * 3; i++)
    {
        res[i] = (unsigned char)(255.0 * (float)src[i]/sumRGB[i/3]);
    }
    return res;
}

__global__ void getP(double* dev_data, double* dev_P_real, double* dev_P_imag, int rows, int cols)
{
    int x       = threadIdx.x + blockIdx.x * blockDim.x;
    int y       = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x < cols) & (y < rows))
    {
        double tempReal1 = 0.0;
        double tempImag1 = 0.0;
        
        double theta1;
        int  pos  = y * cols + x;
        
        for (int ic = 0; ic < cols; ic++)
        {
            theta1 = -2.0 * M_PI * x * ic/(double)cols;
            tempReal1 += dev_data[y * cols + ic] * cos(theta1);
            tempImag1 += dev_data[y * cols + ic] * sin(theta1);
        }

        tempReal1 /= (double)cols;
        tempImag1 /= (double)cols;
        //if (x > 638 & x < 641){
        //    printf("pos : %d, cols : %d, rows: %d, x : %d, y : %d, DIM1 %d, DIM2 %d\n", pos, cols, rows,x,y, gridDim.x, gridDim.y);
        //}
        dev_P_real[pos] = tempReal1;
        dev_P_imag[pos] = tempImag1;
    }       
}
__global__ void REAL_IMG_gpu(double* dev_data, double* dev_res_real, double* dev_res_imag,double* dev_P_real, double* dev_P_imag, int rows, int cols)
{
    int x       = threadIdx.x + blockIdx.x * blockDim.x;
    int y       = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x < cols) & (y < rows))
    {
        double theta2;
        double tempReal2 = 0.0;
        double tempImag2 = 0.0;

        int  pos  = y * cols + x;
        for (int jc = 0; jc < rows; jc++)
        {
            theta2     = -2.0 * M_PI * (y * jc/(double)rows);
            tempReal2 += (cos(theta2) * dev_P_real[jc * cols + x] -
                          sin(theta2) * dev_P_imag[jc * cols + x]);

            tempImag2 += (cos(theta2) * dev_P_imag[jc * cols + x] +
                          sin(theta2) * dev_P_real[jc * cols + x]);

        }
        //if (pos > 200000 & pos < 200002){
        //    printf("pos : %d, cols : %d, rows: %d\n", pos, cols, rows);
        //}
        dev_res_real[pos] = tempReal2/(double)rows;
        dev_res_imag[pos] = tempImag2/(double)rows;
    }    
}

void getComp(double* data, double* res1, double* res2, int width, int height)
{
    double *P_real = new double[width * height];
    double *P_imag = new double[width * height];

    double *dev_data;
    double *dev_res_real ;
    double *dev_res_imag;

    double *dev_P_real;
    double *dev_P_imag;

    int sizeImg = width * height;

    cudaMalloc((void**)&dev_data, sizeImg * sizeof(double));
    cudaMalloc((void**)&dev_res_real , sizeImg * sizeof(double));
    cudaMalloc((void**)&dev_res_imag , sizeImg * sizeof(double));

    cudaMalloc((void**)&dev_P_real , sizeImg * sizeof(double));
    cudaMalloc((void**)&dev_P_imag , sizeImg * sizeof(double));

    cudaMemcpy(dev_data, data, sizeImg * sizeof(double),cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_res_real , res1 , sizeImg * sizeof(double),cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_res_imag , res2 , sizeImg * sizeof(double),cudaMemcpyHostToDevice);    

    cudaMemcpy(dev_P_real , P_real , sizeImg * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_P_imag , P_imag , sizeImg * sizeof(double),cudaMemcpyHostToDevice); 

    int DIM1 = width;
    int DIM2 = height;

    dim3 blocks(DIM2/16 + 1, DIM1/16 + 1);
    dim3 threads(16,16);

    getP<<<blocks, threads>>>(dev_data, dev_P_real, dev_P_imag, DIM1, DIM2);
    REAL_IMG_gpu<<<blocks, threads>>>(dev_data, dev_res_real, dev_res_imag, dev_P_real, dev_P_imag, DIM1, DIM2);

    cudaMemcpy(data, dev_data, sizeImg * sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(res1 , dev_res_real , sizeImg * sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(res2 , dev_res_imag , sizeImg * sizeof(double),cudaMemcpyDeviceToHost);

    cudaFree(dev_data);
    cudaFree(dev_res_real);
    cudaFree(dev_res_imag);
    cudaFree(dev_P_real);
    cudaFree(dev_P_imag);  

    free(P_real);
    free(P_imag);
}
unsigned char* FFT(unsigned char* src, int width, int height)
{
    double* srcF     = new double[width * height];
    double* realComp = new double[width * height];
    double* imagComp = new double[width * height];

    unsigned char* res = new unsigned char[width * height];

    double* resF = new double[width * height];
    for(int i=0; i < width * height; i++)
    {
        srcF[i] = (double)src[i];
    }

    getComp(srcF, realComp, imagComp, width, height);

    double min = 100000000.0;
    double max = -100000000.0;

    for(int i=0; i < width * height; i++)
    {
        //std::cout << "i : " << i <<" ->"<< sqrt(realComp[i] * realComp[i] + imagComp[i] * imagComp[i]) << std::endl;
        resF[i] = (double)sqrt(realComp[i] * realComp[i] + imagComp[i] * imagComp[i]);
        //if (resF[i] < min)
        //    min  = resF[i];
        if (resF[i] > max)
            max  = resF[i];
    }

    for(int i=0; i < width * height; i++)
    {
        resF[i] = 255.0* log10(resF[i] + 1.0)/log10(max + 1.0);
    }

    max = -100000000.0;
    for(int i=0; i < width * height; i++)
    {
        if (resF[i] < min)
        {
            min = resF[i];
        }
        else if (resF[i] > max)
        {
            max = resF[i];
        }
    }
    
    for(int i=0; i < width * height; i++)
    {
        resF[i] = 255.0*(resF[i] - min)/(max - min);
    }
    
    for(int i=0; i < width * height; i++)
    {
        res[i] = (unsigned char)resF[i];
    }

    free(srcF);
    free(realComp);
    free(imagComp);
    free(resF);

    return res;
}
/*
* UTILS
* =========================================================================
*/

double getResizeFactor(int width, int height)
{
    double maxDim = (double)max(width,height);

    double ftr = 480/maxDim;
    ftr = (ftr > 1) ? 1: ftr;
    
    return ftr;
}

void Mat2Mat(cv::Mat& src, cv::Mat& dst, int x0, int y0)
{
    for(int i = x0; i < x0 + src.rows; i++)
    {
        for(int j = y0; j < y0 + src.cols; j++)
        {
            dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(i-x0, j-y0);
        }
    }
}

void copy(unsigned char* src, unsigned char* dst, int size)
{
    for(int i = 0; i < size; i++)
    {
        dst[i] = src[i];
    }
}

cv::Mat fftSwap(cv::Mat src, int cols, int rows)
{

    int mcols = (int)(cols/2);
    int mrows = (int)(rows/2);
    cv::Mat dst(cv::Size(cols, rows), CV_8U, cv::Scalar(0));
    for(int i = 0; i < mrows; i++)
    {
        for(int j = 0; j < mcols; j++)
        {
            dst.at<uchar>(i+mrows,j+mcols) = src.at<uchar>(i,j);
            dst.at<uchar>(i,j)             = src.at<uchar>(i+mrows,j+mcols);
        }
    }
    for(int i = 0; i < mrows; i++)
    {
        for(int j = mcols; j < mcols + mcols; j++)
        {
            dst.at<uchar>(i+mrows,j-mcols) = src.at<uchar>(i,j);
            dst.at<uchar>(i,j)             = src.at<uchar>(i+mrows,j-mcols);
        }
    }

    return dst;
}


unsigned char* BC(unsigned char* src, float B, float C, int size)
{
    float* srcF        = new float[size];
    unsigned char* res = new unsigned char[size];

    for(int i=0; i < size; i++)
    {
        srcF[i] = (float)src[i];
    }

    for(int i=0; i < size; i++)
    {
        srcF[i] = C * srcF[i] + B;
        if (srcF[i] > 255)
            srcF[i] = 255;
    }  

    for(int i=0; i < size; i++)
    {
        res[i] = (unsigned char)srcF[i];
    }
    free(srcF);

    return res;
}

cv::Mat TemplateMatching()
{
    cv::Mat img; 
    cv::Mat templ; 
    cv::Mat result;

    img   = cv::imread("../files/demo.png");
    templ = cv::imread("../files/temp.png");

    cv::Mat img_display;
    img.copyTo( img_display );

    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;

    result.create( result_rows, result_cols, CV_32FC1 );

    cv::matchTemplate( img, templ, result, 1);
    cv::normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    double minVal; 
    double maxVal; 
    
    cv::Point minLoc; 
    cv::Point maxLoc;
    cv::Point matchLoc;

    cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );

    matchLoc = minLoc;
    cv::rectangle( img_display, matchLoc, cv::Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), cv::Scalar::all(0), 2, 8, 0 );
    cv::rectangle( result, matchLoc, cv::Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), cv::Scalar::all(0), 2, 8, 0 );

    return img_display;
}