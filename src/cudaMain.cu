#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <algorithm> 

#define CVUI_IMPLEMENTATION
#include "cvui.h"

#include "cudaMain.h"
#include "tools.h"
#include "base.h"

cv::Mat img;
cv::Mat img_res;
cv::Mat IMAGE;
cv::Mat frame = cv::Mat(720, 1280, CV_8UC3);

int cudaMain(int argc, char **argv)
{
    std::string X;
    
    if (argc == 1)
    {
        X = "files/24BITS.BMP";
        X = "../" + X;

        filename = X.c_str();        
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
                
                filename = X.c_str();
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
            
            filename = X.c_str();        
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
    

    factor = getResizeFactor(WIDTH, HEIGHT);

    cv::resize(img, img_res, cv::Size(), factor, factor, cv::INTER_LINEAR );
    cvui::init(WINDOW_NAME);

    toDisplay = new unsigned char[WIDTH * HEIGHT * 3];
    
    while (true)
    {
        frame = cv::Scalar(38, 36, 26);     
        if (prev_iter != (int)iter)
            DONE = true;
        
        switch (SET_CODE)
        {
            case 0: {
                DONE = true;
                IMAGE = img_res.clone();
                break;
            }
            case 2: {
                if (DONE){
                    copy(mdata, toDisplay, WIDTH*HEIGHT*3);
                    for (int i = 0; i < (int)iter; i++)
                        toDisplay = meanFilter(toDisplay, WIDTH, HEIGHT);
                    cv::Mat img(cv::Size(WIDTH, HEIGHT), CV_8UC3, toDisplay);
                    cv::resize(img, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC );
                    DONE = false;
                    prev_iter = (int)iter;
                }
                break;
            }
            case 4: {
                if (DONE){
                    copy(mdata, toDisplay, WIDTH*HEIGHT*3);
                    toDisplay = laplacianFilter(toDisplay, WIDTH, HEIGHT);
                    cv::Mat img(cv::Size(WIDTH, HEIGHT), CV_8UC3, toDisplay);
                    cv::resize(img, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC );

                    cvtColor(IMAGE, IMAGE, cv::COLOR_RGB2GRAY);
                    cvtColor(IMAGE, IMAGE, cv::COLOR_GRAY2RGB);
                    DONE = false;   
                }
                break;
            }
            case 8: {
                if (DONE) {
                    grayimg = toGray(mdata, WIDTH, HEIGHT);
                    cv::Mat grayCV(cv::Size(WIDTH, HEIGHT), CV_8UC1, grayimg);
                    cvtColor(grayCV, grayCV, cv::COLOR_GRAY2RGB);
                    cv::resize(grayCV, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC );
                    DONE = false; 
                }
                break;
            }
            case 16: {
                if (DONE){
                    chromimg = toChromatic(mdata, WIDTH, HEIGHT);
                    cv::Mat chromCV(cv::Size(WIDTH, HEIGHT), CV_8UC3, chromimg);
                    cv::resize(chromCV, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC );
                    DONE = false;    
                }
                break;
            }
            case 32: {
                if (DONE){
                    grayimg = toGray(mdata, WIDTH, HEIGHT);
                    img2fft = FFT(grayimg, WIDTH, HEIGHT); 
                    cv::Mat grayCV(cv::Size(WIDTH, HEIGHT), CV_8U, img2fft);

                    cv::Mat swapGray = fftSwap(grayCV, WIDTH, HEIGHT);
                    cvtColor(swapGray, swapGray, cv::COLOR_GRAY2RGB);
                    cv::resize(swapGray, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC );
                    DONE = false; 
                }
                break;
            }
        }

        cvui::window(frame, 10, 10, 180, 480, "Settings");
        cvui::window(frame, 720 , 10, 520, 560, "Picture");
        
        if (cvui::checkbox(frame, 15, 35, "Mean Filter", &USE_MEAN_FILTER)){
            USE_LAPLACIAN_FILTER = false;
            USE_CHROMATIC = false;
            USE_GRAY = false;
            USE_FFT  = false;
        } 
        if (cvui::checkbox(frame, 15, 55, "Laplacian Filter", &USE_LAPLACIAN_FILTER)){
            USE_MEAN_FILTER = false;
            USE_CHROMATIC = false;
            USE_GRAY = false;
            USE_FFT  = false;
        }
        if (cvui::checkbox(frame, 15, 75, "Gray Scale", &USE_GRAY)){
            USE_MEAN_FILTER = false;
            USE_LAPLACIAN_FILTER = false;
            USE_CHROMATIC = false;
            USE_FFT  = false;
        }

        if (cvui::checkbox(frame, 15, 95, "Chromatic", &USE_CHROMATIC)){
            USE_MEAN_FILTER = false;
            USE_LAPLACIAN_FILTER = false;
            USE_GRAY = false;
            USE_FFT  = false;
        }

        if (cvui::checkbox(frame, 15, 115, "FFT", &USE_FFT)){
            USE_MEAN_FILTER = false;
            USE_LAPLACIAN_FILTER = false;
            USE_GRAY = false;
            USE_CHROMATIC = false;
        }

        SET_CODE =  USE_MEAN_FILTER      * 2 + 
                    USE_LAPLACIAN_FILTER * 4 +
                    USE_GRAY             * 8 +
                    USE_CHROMATIC        * 16 +
                    USE_FFT * 32;

        cvui::trackbar(frame, 828, 500, 300, &iter, 0.0, 20.0, 0.1, "",cvui::TRACKBAR_HIDE_LABELS); //"%1.Lf"
        cvui::printf(frame, 760, 520, 0.4, 0xeeeeee, "Mean Filter");

        if(cvui::button(frame, 10, 680, "&Quit")){
            break;
        }

        cvui::update();
        int xpos =  (int)((520 - IMAGE.cols)/2) + 720;
        int ypos =  50+20;
        Mat2Mat(IMAGE, frame, ypos, xpos);
        cv::imshow(WINDOW_NAME, frame);
        
        k = cv::waitKey(1);
        if (k == 27){
            std::cout << "[ESC] : break" << std::endl;
            break;
        }
    }  

    free(toDisplay);
    free(mdata);
    
    return 0;   
}