#ifndef __PIXELDIS_H__
#define __PIXELDIS_H__

#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <windows.h>
#include <time.h>

#include <emmintrin.h>
#include <xmmintrin.h>


#include "colourtransform.h"
#include "radiotransform.h"
#include "differencemeasure.h"
#include "aggregation.h"

//#include <omp.h>

//#define PRINT_TIMINGS

class CDsi
{
protected:
	int imgW;
	int imgH; 
	int maxdisp; 
	int border_costs;
	int pixdisthresh;
	int useBT;
	int numthreads;

	int colour_space;
	int radiotransf;
	int difmeasure_method;

	bool leftisReference;
	IplImage *left, *right;

	int *dsi;
	// for view change
	int *dsi_right, *dsi_left;

	void match_scanlines (float *left_data, float *right_data, int *dsi_data);

	float *scale_up (float *scanline);

	void change_view (int start_sl, int end_sl);

	//int *ranktransform (int *input, int winsize);

public:

	CDsi (int _imgW, int _imgH, int _maxdisp, int _numthreads);

	void Generate_DSI (IplImage *_left, IplImage *_right, bool _leftisReference, int _colour_space, int _radiotransf, int _difmeasure_method, int _border_costs, int _pixdisthresh, int _useBT, int *_dsi);

	void DSI_Left_View (int *_dsi_right, int *_dsi_left);

};

//#define SHOW_HMI_STEPS

class CHmiDsi : public CDsi
{
protected:

public:

	CHmiDsi (int _imgW, int _imgH, int _maxdisp, int _numthreads) : CDsi (_imgW, _imgH, _maxdisp, _numthreads) {};

	void WarpToRightView (float *left0, IplImage *disp0, IplImage *occmask0, float *warped0, IplImage *occmask0_right_view);

	void ComputeMiScores (float *right0, float *warped0, IplImage *occmask0_right_view, IplImage **HMI_Scores_Lookup);

	void FillDSI_right_reference (float *right, float *left, IplImage **HMI_Scores_Lookup, int y_start, int y_end, int *dsi);

	void FillDSI_left_reference (float *left, float *right, IplImage **HMI_Scores_Lookup, int y, int *dsi);

	void Generate_DSI (IplImage *_left, IplImage *_right, IplImage *left0, IplImage *right0, IplImage *disp0, IplImage *occmask0, bool _leftisReference, int _colour_space, int _bordercosts, int *_dsi);
};

void comp_dsi (IplImage *left_img, IplImage *right_img, int maxdisp, int winSize, uchar *&dsi, int &dsi_width, int &dsi_height, int bordercosts, int pixdisthresh, int useBT, int scale);

#endif