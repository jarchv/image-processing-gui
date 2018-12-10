#ifndef BASE_H
#define BASE_H

int WIDTH  = 0;
int HEIGHT = 0;

unsigned char* mdata;
const char* filename;
double iter               = 1;
int prev_iter             = (int)iter;
bool USE_MEAN_FILTER      = false;
bool USE_LAPLACIAN_FILTER = false;
bool USE_CHROMATIC        = false;
bool USE_GRAY             = false;
bool DONE                 = true;
   

   
char SET_CODE = USE_MEAN_FILTER * 2 + 
                    USE_LAPLACIAN_FILTER * 4 +
                    USE_GRAY * 8 +
                    USE_CHROMATIC * 16;

double factor;
unsigned char* toDisplay;
unsigned char* grayimg;
unsigned char* chromimg;
char k;
#endif