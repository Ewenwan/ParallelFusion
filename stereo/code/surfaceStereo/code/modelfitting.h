#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "segment.h"
#include "surfacemodels.h"

void FitConstantPlaneToSegment (Segment &segment, int isReference, IplImage *disp, float scale);

void FitPlaneToSegment_ROBUST (Segment &segment, int isReference, IplImage *disp, float scale, float plane_threshold);

IplImage *PlotDisparityPlanes (vector<Segment> segments, int imgW, int imgH, float scale);

void FitSplineToSegment (Segment &segment, IplImage *disp, float scale, int splinenumber);