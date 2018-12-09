#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <algorithm> 

#define CVUI_IMPLEMENTATION
#include "cvui.h"

#include "cudaMain.h"
#include "tools.h"

int WIDTH  = 0;
int HEIGHT = 0;

cv::Mat img;
cv::Mat img_res;

unsigned char* mdata;

/*
void Mat2Array(cv::Mat src, unsigned char* dst, int cols, int rows)
{
    int i, j;
    for(i=0; i < rows; i++)
    {
        for(j=0; j < cols; j++)
        {
            dst[i * cols * 3 + j*3    ] = src.at<cv::Vec3b>(i,j)[0];
            dst[i * cols * 3 + j*3 + 1] = src.at<cv::Vec3b>(i,j)[1];
            dst[i * cols * 3 + j*3 + 2] = src.at<cv::Vec3b>(i,j)[2];
        }
    }
    std::cout<<"end: "<< (i-1) * cols * 3 + (j-1)*3 + 2 << ", " << rows * cols * 3 << std::endl;
}
*/
double getResizeFactor(int width, int height)
{
    double maxDim = (double)max(WIDTH,HEIGHT);

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
int cudaMain(int argc, char **argv)
{
    std::string X;
    
    if (argc == 1)
    {
        X = "files/24BITS.BMP";
        X = "../" + X; 
        const char* filename = X.c_str();        
        std::cout<<"\nFILE: "<<X<<"\n"<<std::endl;
        mdata = readBMPFile(filename, WIDTH, HEIGHT);
        img   = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, mdata);
    }
    else if (argc > 1)
    {        
        if (argc > 2)
        {
            if (argv[1][1] == 't')
            {
                X = argv[2];
                X = "../" + X; 
                const char* filename = X.c_str();
                std::cout<<"\nFILE: "<<X<<"\n"<<std::endl;
                img     = cv::imread(filename,cv::IMREAD_COLOR );
                WIDTH   = (int)img.cols;
                HEIGHT  = (int)img.rows;
                mdata   = new unsigned char[WIDTH * HEIGHT * 3];
                mdata   = img.data;   
            }
        }
        else {
            X = argv[1];
            X = "../" + X; 
            const char* filename = X.c_str();        
            std::cout<<"\nFILE: "<<X<<"\n"<<std::endl;
            mdata = readBMPFile(filename, WIDTH, HEIGHT);
            img   = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, mdata);
        }

    }

    else {
        printf("Unable to open file ...");
        return 0;
    }

    printf("Size = (%d, %d)\n", WIDTH,HEIGHT);
    
    cv::namedWindow(WINDOW_NAME);
    

    double factor = getResizeFactor(WIDTH, HEIGHT);

    cv::resize(img, img_res, cv::Size(), factor, factor, cv::INTER_LINEAR );
    cvui::init(WINDOW_NAME);

    cv::Mat frame = cv::Mat(720, 1280, CV_8UC3);

    
    double iter          = 1;
    int prev_iter        = (int)iter;
    bool USE_MEAN_FILTER = false;
    bool USE_LAPLACIAN_FILTER = false;
    bool DONE            = true;
    
    cv::Mat IMAGE;
    
    unsigned char* toDisplay = new unsigned char[WIDTH * HEIGHT * 3];

    while (true)
    {
        if (prev_iter != (int)iter)
            DONE = true;
        frame = cv::Scalar(38, 36, 26);     
        if (USE_MEAN_FILTER) {
            if (DONE){
                copy(mdata, toDisplay, WIDTH*HEIGHT*3);
                for (int i = 0; i < (int)iter; i++)
                    toDisplay = meanFilter(toDisplay, WIDTH, HEIGHT);
                cv::Mat img(cv::Size(WIDTH, HEIGHT), CV_8UC3, toDisplay);
                cv::resize(img, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC );
                DONE = false;
                prev_iter = (int)iter;
                std::cout << "Done!" << std::endl;
            }
        } else {
            DONE = true;
            IMAGE = img_res.clone();
        }
        cvui::window(frame, 10, 10, 180, 480, "Settings");
        cvui::checkbox(frame, 15, 35, "Mean Filter", &USE_MEAN_FILTER);
        cvui::checkbox(frame, 15, 55, "Laplacian Filter", &USE_LAPLACIAN_FILTER);
        cvui::trackbar(frame, 788, 600, 300, &iter, 0.0, 10.0, 0.1, "",cvui::TRACKBAR_HIDE_LABELS);
        if(cvui::button(frame, 10, 680, "&Quit")){
            break;
        }
        cvui::update();

        Mat2Mat(IMAGE, frame, 80, 640);
        
        cv::imshow(WINDOW_NAME, frame);
        char k = cv::waitKey(1);

        if (k == 27){
            std::cout << "[ESC] : break" << std::endl;
            break;
        }
    }  
    free(toDisplay);
    free(mdata);
    return 0;   
}