#include <stdlib.h>
#include <stdio.h>

#include "cv.h"
#include "highgui.h"

//#include "types.h"

#define SIMPLETREE_PATH "..\\simpletree\\"
#define SIMPLETREE_EXE "..\\simpletree\\dynprog.exe"

IplImage *ExternDisparityMap (char *imgname, IplImage *left, IplImage *right, int maxdisp, float scale, int ST_P1, int ST_P2);
