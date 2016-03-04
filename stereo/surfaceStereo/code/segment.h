#ifndef _SEGMENT_H_
#define _SEGMENT_H_

#include "cv.h"
#include "highgui.h"
#include "surfacemodels.h"

#include <vector>
#include <algorithm>
using namespace std;

#define EDISON_PATH "..\\edison\\"
#define EDISON_EXE "..\\edison\\edison.exe ..\\edison\\config.txt"

class Segment
{
public:

	int segid;

	vector <CvPoint> segpoints;

	// the segments original plane (after plane fitting)*/
	SurfaceModel *surfacemodel;
};


class Segments
{
public:

	vector <Segment> segments;

	Segments ();

	IplImage *InitFromMeanShift (char *imgname, IplImage *img, float SpatialBandwidth, float RangeBandwidth, float MinimumRegionsArea);

	void InitFromSegmentedImage (IplImage *segimage);

	void Plot (IplImage *dst);

	Segment at (unsigned int i) {return segments[i];}

	int size () {return segments.size();}

	void ReadSegmentsFromLabelledImage (IplImage *labelled);

private:

	vector <CvPoint> connectivity;
};



#endif

