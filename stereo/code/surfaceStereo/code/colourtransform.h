#ifndef __COLOURTRANSFORM_H__
#define __COLOURTRANSFORM_H__

#include <stdio.h>
#include <algorithm>
#include "opencv/cv.h"
#include "opencv/highgui.h"

#define GREY 0
#define RGB 1
#define I1I2I3 2
#define AC1C2 3
#define YC1C2 4
#define H1H2H3 5
#define XYZ 6
#define LUV 7
#define LAB 8
#define HSI 9
#define LUV_SCALED 10
#define LAB_SCALED 11
#define LUV_SCALED5 12
#define LUV_SCALED10 13
#define CMY 14

float *colour_transform(IplImage *input, int color_model);

#endif