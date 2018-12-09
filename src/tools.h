#ifndef TOOLS_H
#define TOOLS_H

unsigned char* readBMPFile( char const*  filename,
    int& width,
    int& height);

unsigned char *meanFilter(unsigned char* dev_src, 
    int height, 
    int width);
unsigned char * FilterOp(unsigned char* data,  int height, int width, float* kernel);
#endif