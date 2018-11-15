#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <iostream>
#include "tools.cuh"

using namespace cv;
using namespace std;

constexpr unsigned int  str2int(const char* str,  int h = 0)
{
    return !str[h] ? 5381: (str2int(str, h + 1) * 33) ^ str[h];
}

int main(int argc, char* argv[])
{
    string X;
    bool    THRESHOLD = false;
    bool    GRAY      = false;
    bool    HISTOGRAM = false;    
    int     limit;

    if (argc > 1)
    {
        for (int i = 0; i < argc; i++)
        {
            printf("Arg %d : %s\n", i, argv[i]);
            
            switch (str2int(argv[i]))
            {
                case str2int("-thr"):
                    if (i + 1 < argc)
                    {
                        THRESHOLD = true;
                        GRAY      = false;
                        limit     = strtol(argv[i+1], NULL, 10);
                    }                      
                    break;

                case str2int("-gray"):
                    GRAY      = true;
                    THRESHOLD = false;

                case str2int("-hist"):
                    HISTOGRAM = true;
                
                default:
                    break;
            }
        }
        
        X = argv[1];
        X = "bmpreader/" + X;
    }

    else {
        printf("Unable to open file ...");
        return 0;
    }

    cout<<"\nFILE: "<<X<<"\n"<<endl;
    
    if (THRESHOLD)
        printf("\tTHRESHOLD LIMIT = %d\n\n", limit);
    if (GRAY)
        printf("TO GRAYSCALE...\n\n");
    if (HISTOGRAM)
        printf("GETTING HISTOGRAM...\n\n");

    const char* filename = X.c_str();
    unsigned char* mdata;
    
    int W, H, D = 24;
     
    mdata = readBMPFile(filename, W, H, D);
    
    namedWindow("foo");    
    if (D < 24){
        Mat img(Size(W, H), CV_8U, mdata);
        imshow("foo", img);
    }
    else {    
        Mat img(Size(W, H), CV_8UC3, mdata);
        imshow("foo", img);
    }
    waitKey(0);    
            
    return 0;   
}
