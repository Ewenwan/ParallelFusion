#ifndef WARP_H
#define WARP_H

#include "opencv/cv.h"
#include "opencv/highgui.h"

IplImage *Occ_Mask_Via_Disp_Map (IplImage *scaled_disp, float scale);

IplImage *Unscale_Image (IplImage *src, float scale);

#endif