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
cv::Mat templ;
cv::Mat templ_res  = cv::Mat(60, 60, CV_8UC3,cv::Scalar(38, 36, 26));
cv::Mat chromCV;
cv::Mat grayCV;
cv::Mat swapGray;
cv::Mat trackResul;
cv::Mat TRACK;
cv::Mat RESULT;
cv::Mat frameCap;

int cudaMain(int argc, char **argv)
{
    std::string X;
    
    if (argc == 1)
    {
        X = "files/24BITS.BMP";
        X = "../" + X;

        filename = X.c_str();        
        std::cout<<"\nFILE: "<<X<<"\n"<<std::endl;
        
        mdata = readBMPFile(filename, WIDTH, HEIGHT, DEPTH);
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
                //mdata   = img.data;
                mdata   = toArray(img);  
            }
        }
        else {
            X = argv[1];
            X = "../" + X; 
            
            filename = X.c_str();        
            std::cout<<"\nFILE: "<<X<<"\n"<<std::endl;
            mdata = readBMPFile(filename, WIDTH, HEIGHT, DEPTH);
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
        cvui::window(frame,  10, 10, 260, 280, "Settings");
        cvui::window(frame, 720, 10, 520, 680, "Picture");
        cvui::window(frame, 620, 10,  80, 100, "Template");

        if (prev_iter != (int)iter)
            DONE = true;
        
        if (prev_brig != brig)
            DONE = true;

        if (prev_cont != cont)
            DONE = true; 

        switch (SET_CODE)
        {
            case 0: {
                //img = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, mdata);
                //if (max(img.cols, img.rows) > MAX_DIM)
                //    cv::resize(img, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC);
                //else
                    
                img_res.copyTo(IMAGE);
                DONE = true;       
                break;
            }
            case 2: {
                if (DONE){

                    copy(mdata, toDisplay, WIDTH*HEIGHT*3);
                    for (int i = 0; i < (int)iter; i++)
                        toDisplay = meanFilter(toDisplay, WIDTH, HEIGHT);
                    img = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, toDisplay);
                    if (max(img.cols, img.rows) > MAX_DIM)
                        cv::resize(img, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC);
                    else
                        img.copyTo(IMAGE);
                    DONE = false;
                    prev_iter = (int)iter;
                }
                break;
            }
            case 4: {
                if (DONE){
                    
                    copy(mdata, toDisplay, WIDTH*HEIGHT*3);
                    toDisplay = laplacianFilter(toDisplay, WIDTH, HEIGHT);
                    img = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, toDisplay);
                    if (max(img.cols, img.rows) > MAX_DIM)
                        cv::resize(img, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC);
                    else
                        img.copyTo(IMAGE);

                    cvtColor(IMAGE, IMAGE, cv::COLOR_RGB2GRAY);
                    cvtColor(IMAGE, IMAGE, cv::COLOR_GRAY2RGB);
                    DONE = false;   
                }
                break;
            }
            case 8: {
                if (DONE) {
                    grayimg = toGray(mdata, WIDTH, HEIGHT);
                    grayCV  = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC1, grayimg);
                    cvtColor(grayCV, grayCV, cv::COLOR_GRAY2RGB);
                    if (max(grayCV.cols, grayCV.rows) > MAX_DIM)
                        cv::resize(grayCV, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC);
                    else
                        grayCV.copyTo(IMAGE);
                    DONE = false; 
                }
                break;
            }
            case 16: {
                if (DONE){
                    chromimg = toChromatic(mdata, WIDTH, HEIGHT);
                    chromCV  = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, chromimg);
                    if (max(chromCV.cols, chromCV.rows) > MAX_DIM)
                        cv::resize(chromCV, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC);
                    else
                        chromCV.copyTo(IMAGE);
                    DONE = false;
                    prev_brig = brig;
                    prev_cont = cont;   
                }
                break;
            }
            case 32: {
                if (DONE){
                    grayimg = toGray(mdata, WIDTH, HEIGHT);
                    img2fft = FFT(grayimg, HEIGHT, WIDTH);
                    imgBC   = BC(img2fft, (float)brig, cont, WIDTH * HEIGHT);

                    grayCV   = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8U, imgBC);
                    swapGray = fftSwap(grayCV, WIDTH, HEIGHT);
                    cvtColor(swapGray, swapGray, cv::COLOR_GRAY2RGB);

                    if (max(swapGray.cols, swapGray.rows) > MAX_DIM)
                        cv::resize(swapGray, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC);
                    else
                        swapGray.copyTo(IMAGE);
                    prev_brig = brig;
                    prev_cont = cont; 
                    DONE = false;
                }
                break;
            }

            case 64: {
                if (DONE){
                    templ    = cv::imread("../files/temp.png");
                    cv::VideoCapture cap("../files/videodemo.mp4");

                    if (!cap.isOpened())
                    {
                        std::cout << "Failed to open camera." << std::endl;
                    }          

                    else {
                        cap >> frameCap;
                        factor = getResizeFactor(frameCap.cols, frameCap.rows);

                        int count = 0;
                        for(;;)
                        {
                            cvui::checkbox(frame, 15, 135, "Template Matching   ", &USE_TEMPLATE);
                            cap >> frameCap;
                            if(frameCap.empty())
                                break;
                            count++;
                            if(count > 128)
                                break;
                            if(USE_TEMPLATE == false)
                                break;

                            trackResul = TemplateMatching(frameCap, TRACK, templ);
                            cvtColor(trackResul, trackResul, cv::COLOR_GRAY2RGB);
                            cv::normalize( trackResul, trackResul, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );
                            trackResul.convertTo(trackResul, CV_8U);
                            if (max(TRACK.cols, TRACK.rows) > MAX_DIM){
                                cv::resize(TRACK, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC);
                                cv::resize(trackResul, RESULT, cv::Size(), factor, factor, cv::INTER_CUBIC);
                            }
                            else{
                                TRACK.copyTo(IMAGE);
                                trackResul.copyTo(RESULT);
                            }

                            cv::resize(templ, templ_res, cv::Size(60,60), cv::INTER_CUBIC);
                            int xpos =  (int)((520 - IMAGE.cols)/2) + 720;
                            int ypos =  50+0;
                            Mat2Mat(IMAGE , frame, ypos, xpos);
                            Mat2Mat(RESULT, frame, ypos+300, xpos); 
                            Mat2Mat(templ_res, frame, 40, 630);

                            cv::imshow(WINDOW_NAME, frame);
                            k = cv::waitKey(1);
                            if (k == 27){
                                std::cout << "[ESC] : break" << std::endl;
                                break;
                            }                           
                        }
                        cap.release();                        
                    }

                    DONE = false;
                }
                factor = getResizeFactor(img.cols, img.rows);
                cvui::checkbox(frame, 15, 135, "Template Matching   ", &USE_TEMPLATE);
                break;
            }
            default:{
                DONE = true;
            }
        }

        

        Mat2Mat(templ_res, frame, 40, 630);

        if (cvui::checkbox(frame, 15, 35, "Mean Filter", &USE_MEAN_FILTER)){
            USE_LAPLACIAN_FILTER = false;
            USE_CHROMATIC = false;
            USE_GRAY = false;
            USE_FFT  = false;
            USE_TEMPLATE = false;
        } 
        if (cvui::checkbox(frame, 15, 55, "Laplacian Filter", &USE_LAPLACIAN_FILTER)){
            USE_MEAN_FILTER = false;
            USE_CHROMATIC = false;
            USE_GRAY = false;
            USE_FFT  = false;
            USE_TEMPLATE = false;
        }
        if (cvui::checkbox(frame, 15, 75, "Gray Scale", &USE_GRAY)){
            USE_MEAN_FILTER = false;
            USE_LAPLACIAN_FILTER = false;
            USE_CHROMATIC = false;
            USE_FFT  = false;
            USE_TEMPLATE = false;
        }

        if (cvui::checkbox(frame, 15, 95, "Chromatic", &USE_CHROMATIC)){
            USE_MEAN_FILTER = false;
            USE_LAPLACIAN_FILTER = false;
            USE_GRAY = false;
            USE_FFT  = false;
            USE_TEMPLATE = false;
        }

        if (cvui::checkbox(frame, 15, 115, "Fourier", &USE_FFT)){
            USE_MEAN_FILTER = false;
            USE_LAPLACIAN_FILTER = false;
            USE_GRAY = false;
            USE_CHROMATIC = false;
            USE_TEMPLATE = false;
        }

        if (cvui::checkbox(frame, 15, 135, "Template Matching   ", &USE_TEMPLATE)){
            USE_MEAN_FILTER = false;
            USE_LAPLACIAN_FILTER = false;
            USE_GRAY = false;
            USE_CHROMATIC = false;
            USE_FFT = false;
        }

        SET_CODE =  USE_MEAN_FILTER      *  2 + 
                    USE_LAPLACIAN_FILTER *  4 +
                    USE_GRAY             *  8 +
                    USE_CHROMATIC        * 16 +
                    USE_FFT              * 32 +
                    USE_TEMPLATE         * 64;

        cvui::trackbar(frame, 828, 550, 300, &brig,   0,  255,   1, "",cvui::TRACKBAR_HIDE_LABELS);
        cvui::trackbar(frame, 828, 585, 300, &cont, 0.1, 100.0, 0.1, "",cvui::TRACKBAR_HIDE_LABELS);
        cvui::trackbar(frame, 828, 615, 300, &iter, 0.0, 20.0, 0.1, "",cvui::TRACKBAR_HIDE_LABELS); //"%1.Lf"

        cvui::printf(frame, 760, 570, 0.4, 0xeeeeee, "Brightness");
        cvui::printf(frame, 760, 605, 0.4, 0xeeeeee, "Contrast");
        cvui::printf(frame, 760, 635, 0.4, 0xeeeeee, "Mean Filter");
        cvui::printf(frame,  20, 300, 0.4, 0xeeeeee, "Filename : %s", filename);
        cvui::printf(frame,  20, 320, 0.4, 0xeeeeee, "Width  : %d", WIDTH);
        cvui::printf(frame,  20, 340, 0.4, 0xeeeeee, "Height : %d", HEIGHT);
        cvui::printf(frame,  20, 360, 0.4, 0xeeeeee, "Depth  : %d", DEPTH);
        if(cvui::button(frame, 10, 680, "&Quit")){
            break;
        }

        cvui::update();
        int xpos =  (int)((520 - IMAGE.cols)/2) + 720;
        int ypos =  50+0;
        Mat2Mat(IMAGE, frame, ypos, xpos);
        cv::imshow(WINDOW_NAME, frame);
        
        k = cv::waitKey(1);
        if (k == 27){
            std::cout << "[ESC] : break" << std::endl;
            break;
        }
    }  

    free(toDisplay);
    //free(mdata);
    free(grayimg);
    free(chromimg);
    free(img2fft);
    free(imgBC);

    return 0;   
}