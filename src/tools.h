#ifndef TOOLS_H
#define TOOLS_H

unsigned char* readBMPFile(
    char const*  filename,
    int& width,
    int& height);

unsigned char *meanFilter(
    unsigned char* dev_src, 
    int height, 
    int width);

unsigned char *laplacianFilter(
    unsigned char* data, 
    int height, 
    int width);

unsigned char * FilterOp(
    unsigned char* data,
    int height,
    int width,
    float* kernel);

float * getSumRGB(unsigned char* src, int width, int height);
unsigned char* toGray(unsigned char* src, int with, int height);
unsigned char* toChromatic(unsigned char* src, int width, int height);
#endif