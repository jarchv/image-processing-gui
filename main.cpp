#include <stdio.h>
#include <stdlib.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;
unsigned char* readBMPFile( char const*  filename,
                            int& width,
                            int& height,
                            int& depth)
{
    int             i;
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
    
    int row_padding = width * pad;
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < row_padding; j++)
        {
            res[i*row_padding + j] = data[(height - i)*row_padding + j];
        }
    }  
    
    return res;
}

int main(int argc, char** argv)
{
    string X;
    if (argc > 1)
    {
        X = argv[1];
        X = "bmpreader/" + X;
    }
    
    cout<<"\nFILE: "<<X<<"\n"<<endl;
    
    const char* filename = X.c_str();
    unsigned char* mdata;
    
    int W, H, D = 24;
     
    mdata = readBMPFile(filename, W, H, D);
    
    long numColors = 1 << D;
    
    namedWindow("foo");    
    if (D < 24){
        Mat img(Size(W, H), CV_8U, mdata);
        Mat dst;
        img.convertTo(dst, CV_8UC3);
        imshow("foo", img);
    }
    else {    
        Mat img(Size(W, H), CV_8UC3, mdata);
        imshow("foo", img);
    }
    waitKey(0);    
            
    return 0;   
}
