#ifndef __DIFFERENCEMEASURE_H__
#define __DIFFERENCEMEASURE_H__

#include <stdio.h>

#include "cv.h"
#include "highgui.h"

#define L1 0
#define L2 1
#define MINIMUM 2
#define MAXIMUM 3
#define MEDIAN 4
#define BELLI 5
#define HSI_DIF 6

float differencemeasure (float *ref, float *match, int difmeasure_method);

#endif