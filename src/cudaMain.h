#ifndef CUDAMAIN_H
#define CUDAMAIN_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define WINDOW_NAME "foo"

int    cudaMain(int argc, char **argv);
void   Mat2Mat(cv::Mat& src, cv::Mat& dst, int x0, int y0);
double getResizeFactor(int width, int height);
void   to2d(unsigned char *src, unsigned char**dst, int w, int h, int c);

#endif