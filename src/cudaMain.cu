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
cv::Mat inputbox;
cv::Mat imgFrame;

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
                X = "../files/" + X; 
                
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
            X = "../files/" + X; 
            
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
    

    factor = getResizeFactor(WIDTH, HEIGHT, 480);


    cv::resize(img, img_res, cv::Size(), factor, factor, cv::INTER_LINEAR );
    cvui::init(WINDOW_NAME);

    toDisplay = new unsigned char[WIDTH * HEIGHT * 3];
    
    cv::VideoWriter video("out.avi", CV_FOURCC('M','J','P','G'),30, cv::Size(1280,720),true);

    while (true)
    {
        frame = cv::Scalar(38, 36, 26);     
        cvui::window(frame,  10,  10, 260, 330, "Settings");
        cvui::window(frame, 720,  10, 520, 680, "Result");
        cvui::window(frame, 400, 370, 300, 320, "Input");
        
        if (prev_iter != (int)iter)
            DONE = true;
        
        if (prev_brig != brig)
            DONE = true;

        if (prev_cont != cont)
            DONE = true; 

        switch (SET_CODE)
        {
            case 0: {
                img_res.copyTo(IMAGE);
                DONE = true;       
                break;
            }
            case 2: {
                if (DONE){
                    copy(mdata, toDisplay, WIDTH*HEIGHT*3);
                    for (int i = 0; i < (int)iter; i++)
                        toDisplay = meanFilter(toDisplay, WIDTH, HEIGHT);
                    imgFrame = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, toDisplay);
                    //cvtColor(imgFrame, imgFrame, cv::COLOR_BGR2RGB);
                    if (max(imgFrame.cols, imgFrame.rows) > MAX_DIM)
                        cv::resize(imgFrame, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC);
                    else
                        imgFrame.copyTo(IMAGE);
                    DONE = false;
                    prev_iter = (int)iter;
                }
                break;
            }
            case 4: {
                if (DONE){
                    
                    copy(mdata, toDisplay, WIDTH*HEIGHT*3);
                    toDisplay = laplacianFilter(toDisplay, WIDTH, HEIGHT);
                    imgFrame = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, toDisplay);
                    
                    if (max(imgFrame.cols, imgFrame.rows) > MAX_DIM)
                        cv::resize(imgFrame, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC);
                    else
                        imgFrame.copyTo(IMAGE);

                    cvtColor(IMAGE, IMAGE, cv::COLOR_RGB2GRAY);
                    cvtColor(IMAGE, IMAGE, cv::COLOR_GRAY2RGB);
                    DONE = false;   
                }
                break;
            }

            case 8: {
                if (DONE){
                    copy(mdata, toDisplay, WIDTH*HEIGHT*3);
                    toDisplay = sharpenFilter(toDisplay, WIDTH, HEIGHT);
                    imgFrame = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, toDisplay);
                    //cvtColor(imgFrame, imgFrame, cv::COLOR_BGR2RGB);

                    if (max(imgFrame.cols, imgFrame.rows) > MAX_DIM)
                        cv::resize(imgFrame, IMAGE, cv::Size(), factor, factor, cv::INTER_CUBIC);
                    else
                        imgFrame.copyTo(IMAGE);
                    DONE = false;   
                }
                break;
            }

            case 16: {
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
            case 32: {
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
            case 64: {
                if (DONE){
                    copy(mdata, toDisplay, WIDTH*HEIGHT*3);
                    for (int i = 0; i < (int)iter; i++)
                        toDisplay = meanFilter(toDisplay, WIDTH, HEIGHT);
                    grayimg = toGray(toDisplay, WIDTH, HEIGHT);
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
                    prev_iter = (int)iter;
                    DONE = false;
                }
                break;
            }
            case  128: {
                if (DONE)
                {
                    cv::Mat src;
                    img_res.copyTo(src);

                    std::vector<cv::Mat> bgr(3);
                    split(src,bgr);
                    
                    for (int i = 0; i < 3; i++)
                    {
                        bgr[i].convertTo(bgr[i],CV_32FC1);
                        HaarWavelet(bgr[i],2);

                        cv::normalize(bgr[i], bgr[i], 0, 250, cv::NORM_MINMAX, -1, cv::Mat());
                        bgr[i].convertTo(bgr[i], CV_8U);
                    }

                    cv::merge(bgr, IMAGE);
                    DONE = false;
                }
                break;
            }
            case 256: {
                cvui::window(frame, 620, 10,  80, 100, "Template");
                if (DONE){
                    templ    = cv::imread("../files/temp.png");
                    cv::VideoCapture cap("../files/videodemo.mp4");

                    if (!cap.isOpened())
                    {
                        std::cout << "Failed to open camera." << std::endl;
                    }          

                    else {
                        cap >> frameCap;
                        factor = getResizeFactor(frameCap.cols, frameCap.rows, 480);

                        int count = 0;
                        for(;;)
                        {
                            cvui::checkbox(frame, 30, 295, "Template Matching   ", &USE_TEMPLATE);
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
                            video.write(frame);
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
                factor = getResizeFactor(img.cols, img.rows, 480);
                break;
            }
            default:{
                DONE = true;
            }
        }

        

        //Mat2Mat(templ_res, frame, 40, 630);

        cvui::printf(frame,  20, 45, 0.4, 0xeeeeee, "Filters");

        if (cvui::checkbox(frame, 30, 65, "Mean Filter", &USE_MEAN_FILTER)){
            USE_LAPLACIAN_FILTER = false;
            USE_SHARPEN_FILTER   = false;
            USE_CHROMATIC        = false;
            USE_GRAY             = false;
            USE_FFT              = false;
            USE_TEMPLATE         = false;
            USE_WAVELET          = false;
        } 

        if (cvui::checkbox(frame, 30, 85, "Laplacian Filter", &USE_LAPLACIAN_FILTER)){
            USE_MEAN_FILTER      = false;
            USE_SHARPEN_FILTER   = false;
            USE_CHROMATIC        = false;
            USE_GRAY             = false;
            USE_FFT              = false;
            USE_TEMPLATE         = false;
            USE_WAVELET          = false;  
        }

        if (cvui::checkbox(frame, 30, 105, "Sharpen Filter", &USE_SHARPEN_FILTER)){
            USE_MEAN_FILTER      = false;
            USE_LAPLACIAN_FILTER = false;
            USE_CHROMATIC        = false;
            USE_GRAY             = false;
            USE_FFT              = false;
            USE_TEMPLATE         = false;
            USE_WAVELET          = false;
        }

        cvui::printf(frame,  20, 135, 0.4, 0xeeeeee, "Color Spaces");
        if (cvui::checkbox(frame, 30, 155, "Gray Scale", &USE_GRAY)){
            USE_MEAN_FILTER      = false;
            USE_LAPLACIAN_FILTER = false;
            USE_SHARPEN_FILTER   = false;
            USE_CHROMATIC        = false;
            USE_FFT              = false;
            USE_TEMPLATE         = false;
            USE_WAVELET          = false;
        }

        if (cvui::checkbox(frame, 30, 175, "Chromatic", &USE_CHROMATIC)){
            USE_MEAN_FILTER      = false;
            USE_LAPLACIAN_FILTER = false;
            USE_SHARPEN_FILTER   = false;
            USE_GRAY             = false;
            USE_FFT              = false;
            USE_WAVELET          = false;
            USE_TEMPLATE         = false;            
        }
        
        cvui::printf(frame,  20, 205, 0.4, 0xeeeeee, "Frecuency domain");
        if (cvui::checkbox(frame, 30, 225, "Fourier", &USE_FFT)){
            USE_MEAN_FILTER      = false;
            USE_LAPLACIAN_FILTER = false;
            USE_SHARPEN_FILTER   = false;
            USE_GRAY             = false;
            USE_CHROMATIC        = false;
            USE_WAVELET          = false;
            USE_TEMPLATE         = false;
        }
        if (cvui::checkbox(frame, 30, 245, "Wavelet", &USE_WAVELET)){
            USE_MEAN_FILTER      = false;
            USE_LAPLACIAN_FILTER = false;
            USE_SHARPEN_FILTER   = false;
            USE_GRAY             = false;
            USE_CHROMATIC        = false;
            USE_TEMPLATE         = false;
            USE_FFT              = false;
        }

        cvui::printf(frame,  20, 275, 0.4, 0xeeeeee, "Image Analysis");
        if (cvui::checkbox(frame, 30, 295, "Template Matching   ", &USE_TEMPLATE)){
            USE_MEAN_FILTER      = false;
            USE_LAPLACIAN_FILTER = false;
            USE_SHARPEN_FILTER   = false;
            USE_GRAY             = false;
            USE_CHROMATIC        = false;
            USE_FFT              = false;
            USE_WAVELET          = false;
        }

        SET_CODE =  USE_MEAN_FILTER      *  2  + 
                    USE_LAPLACIAN_FILTER *  4  +
                    USE_SHARPEN_FILTER   *  8  +
                    USE_GRAY             * 16  +
                    USE_CHROMATIC        * 32  +
                    USE_FFT              * 64  +
                    USE_WAVELET          * 128 +
                    USE_TEMPLATE         * 256;

        
        cvui::trackbar(frame, 828, 550, 300, &iter, 0.0, 20.0, 0.1, "",cvui::TRACKBAR_HIDE_LABELS); //"%1.Lf"
        cvui::trackbar(frame, 828, 585, 300, &brig,   0,  255,   1, "",cvui::TRACKBAR_HIDE_LABELS);
        cvui::trackbar(frame, 828, 615, 300, &cont, 0.1, 100.0, 0.1, "",cvui::TRACKBAR_HIDE_LABELS);

        cvui::printf(frame, 760 , 570, 0.4, 0xeeeeee, "Mean Filter");
        cvui::printf(frame, 760 , 605, 0.4, 0xeeeeee, "Brightness");
        cvui::printf(frame, 1120, 605, 0.4, 0xeeeeee, "(Fourier)");
        cvui::printf(frame, 760 , 635, 0.4, 0xeeeeee, "Contrast");
        cvui::printf(frame, 1120, 635, 0.4, 0xeeeeee, "(Fourier)");

        // 570, 605, 635
        cvui::printf(frame,  20, 350, 0.4, 0xeeeeee, "Filename : %s", filename);
        cvui::printf(frame,  20, 370, 0.4, 0xeeeeee, "Width  : %d", WIDTH);
        cvui::printf(frame,  20, 390, 0.4, 0xeeeeee, "Height : %d", HEIGHT);
        cvui::printf(frame,  20, 410, 0.4, 0xeeeeee, "Depth  : %d", DEPTH);

        if(cvui::button(frame, 10, 680, "&Quit")){
            break;
        }

        cvui::update();
        int xpos =  (int)((520 - IMAGE.cols)/2) + 720;
        int ypos =  50+0;
        Mat2Mat(IMAGE, frame, ypos, xpos);

        inputbox_factor = getResizeFactor(WIDTH, HEIGHT, 280);
        cv::resize(img, inputbox, cv::Size(), inputbox_factor, inputbox_factor, cv::INTER_CUBIC);
        Mat2Mat(inputbox, frame, 400, 410);
        
        
        cv::imshow(WINDOW_NAME, frame);
        video.write(frame);
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

    video.release();
    //cvReleaseVideoWriter( &video );
    return 0;   
}