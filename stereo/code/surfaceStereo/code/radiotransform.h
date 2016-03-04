#ifndef __RADIOTRANSFORM_H__
#define __RADIOTRANSFORM_H__

#include <stdio.h>
#include "opencv/cv.h"

#define NO_TRANSFORM 0

#define HMI 1

#define RANK_TRANSFORM_9 10
#define RANK_TRANSFORM_15 11
#define RANK_TRANSFORM_25 12
#define RANK_TRANSFORM_35 12

float *radiotransform (float *input, int imgW, int imgH, int radiotransf);

#endif