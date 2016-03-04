#include "externmatcher.h"

void SplitImage (IplImage *src, IplImage *dst)
{
	int imgW = dst->width;
	int imgH = dst->height;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			dst->imageData[y * dst->widthStep + x] = 
				src->imageData[y * src->widthStep + 3 * (imgW + x)];
		}
	
}

void ConvertToOneChannel (IplImage *three, IplImage *one)
{
	int imgW = three->width;
	int imgH = three->height;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			one->imageData[y * one->widthStep + x] = 
				three->imageData[y * three->widthStep + 3 * x];
		}
}

IplImage *LoadPrecomputed (char *leftname)
{
	IplImage *disp = 0;

	//if (strstr (leftname, "tsukuba"))
	//	disp = cvLoadImage ("simpletree\\precomputed\\tsukuba.pgm");
	//if (strstr (leftname, "venus"))
	//	disp = cvLoadImage ("simpletree\\precomputed\\venus.pgm");
	//if (strstr (leftname, "teddy"))
	//	disp = cvLoadImage ("simpletree\\precomputed\\teddy.pgm");
	//if (strstr (leftname, "cones"))
	//	disp = cvLoadImage ("simpletree\\precomputed\\cones.pgm");

	if (!disp) 
		return 0;

	IplImage *one = cvCreateImage (cvSize (disp->width, disp->height), IPL_DEPTH_8U, 1);
	ConvertToOneChannel (disp, one);
	cvReleaseImage (&disp);
	return one;
}

void Round (IplImage *src, IplImage *dst, float scale)
{
	int imgW = src->width;
	int imgH = src->height;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			float val = (float) (uchar) src->imageData [y * src->widthStep + x];
			val /= scale;
			int rounded = ((int) (val + 0.5f)) * (int) scale;
			dst->imageData [y * dst->widthStep + x] = (char) rounded;
		}
}

IplImage *ExternDisparityMap (char *imgname, IplImage *left, IplImage *right, int maxdisp, float scale, int ST_P1, int ST_P2)
{
	////////////////////////////////////////////////
	// check if we have a precomputed disparity map
	char fn[2048];
	sprintf (fn, "%s\\%s_P1_%d_P2_%d.png", SIMPLETREE_PATH, imgname, ST_P1, ST_P2);

	IplImage *disp = cvLoadImage (fn, 0);
	if (!disp)
	{
		char dummy[512];

		sprintf_s (dummy, sizeof (dummy), "%s\\left.png", SIMPLETREE_PATH); 
		cvSaveImage (dummy, left);
		sprintf_s (dummy, sizeof (dummy), "%s\\right.png", SIMPLETREE_PATH); 
		cvSaveImage (dummy, right);

		sprintf_s (dummy, sizeof (dummy), "%s -calib %s\\calib.txt -d %d -s %.3f -P1 %d -P2 %d -ms 20 %s\\left.png %s\\right.png %s", SIMPLETREE_EXE, SIMPLETREE_PATH, maxdisp, scale, ST_P1, ST_P2, SIMPLETREE_PATH, SIMPLETREE_PATH, fn); 
		system(dummy);

		disp = cvLoadImage (fn, 0);
	}

	// median filter
	IplImage *smoothed = cvCloneImage (disp);
	cvSmooth (disp, smoothed, CV_MEDIAN);

	cvReleaseImage (&disp);

	disp = smoothed;

	return disp;
}