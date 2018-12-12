#ifndef BASE_H
#define BASE_H

int WIDTH  = 0;
int HEIGHT = 0;
int DEPTH  = 24;

const char* filename;
double iter               = 0;
int   brig                = 80;
int   prev_brig           = brig;
double cont               = 10.0;
double prev_cont          = cont;    
int prev_iter             = (int)iter;
bool USE_MEAN_FILTER      = false;
bool USE_LAPLACIAN_FILTER = false;
bool USE_CHROMATIC        = false;
bool USE_GRAY             = false;
bool USE_FFT              = false;
bool USE_TEMPLATE         = false;  
bool DONE                 = true;
   
double inputbox_factor;
   
int SET_CODE = USE_MEAN_FILTER      *  2 + 
                USE_LAPLACIAN_FILTER *  4 +
                USE_GRAY             *  8 +
                USE_CHROMATIC        * 16 +
                USE_FFT              * 32 +
                USE_TEMPLATE         * 64;

double factor;

unsigned char* mdata;
unsigned char* toDisplay;
unsigned char* grayimg;
unsigned char* chromimg;
unsigned char* img2fft;
unsigned char *imgBC;

int MAX_DIM = 480;
char k;
#endif