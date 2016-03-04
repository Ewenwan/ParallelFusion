#include "fusion.h"

#define LARGEVAL 99999

////////////////////////////////////
//	data term without occlusion handling (not used in paper)
void AddUnaries (QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, int *dsi)
{
	int imgW = proposal1.imgW;
	int imgH = proposal1.imgH;

	float scale = proposal1.scale;
	int maxdisp = proposal1.maxdisp;

	q->AddNode(imgW * imgH); 

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			CvPoint p = cvPoint (x,y);

			int c1 = MatchingCosts (p, proposal1, dsi, LARGEVAL);
			int c2 = MatchingCosts (p, proposal2, dsi, LARGEVAL);

			q->AddUnaryTerm(y * imgW + x, c1, c2);


		}
}

int EvaluateUnaryCosts (Proposal &solution, int *dsi)
{
	int imgW = solution.imgW;
	int imgH = solution.imgH;

	float scale = solution.scale;
	int maxdisp = solution.maxdisp;	

	int e_data = 0;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			CvPoint p = cvPoint (x,y);

			int c = MatchingCosts (p, solution, dsi, LARGEVAL);
			e_data += c;
		}

	return e_data;
}
////////////////////////////////////


/////////////////////////////////////
// the penalty on curvature (eq. (8)
// this is a simple unary potential

// computes overall penatly given curvature
int ComputeCurvPen (float Curv, int CurvPen)
{
	float ScaledPen = ((float) CurvPen) / 10.f;
	float Pen = Curv * ScaledPen;

	return (int) (Pen + 0.5f);
}

void AddCurvatureUnary (QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, int CurvPen)
{
	int imgW = proposal1.imgW;
	int imgH = proposal1.imgH;

	for (int y = CURVRAD; y < imgH - CURVRAD; y++)
		for (int x = CURVRAD; x < imgW - CURVRAD; x++)
		{
			SurfaceModel *sm = proposal1.surfacemodels[y * imgW + x];

			int E0 = 0;
			// if the current surface is a spline compute its curvature at this point
			if (sm->which_type() == IS_SPLINE)
			{
				BSpline *spline = (BSpline*) sm;
				E0 = ComputeCurvPen (spline->PointCurvature (cvPoint (x,y)), CurvPen);
			}

			sm = proposal2.surfacemodels[y * imgW + x];

			int E1 = 0;
			if (sm->which_type() == IS_SPLINE)
			{
				BSpline *spline = (BSpline*) sm;
				E1 = ComputeCurvPen (spline->PointCurvature (cvPoint (x,y)), CurvPen);
			}

			// add the curvature unary
			q->AddUnaryTerm(y * imgW + x, E0, E1);
		}
}

// computes costs due to curvature term
int EvaluateCurvatureUnary (Proposal &solution, int CurvPen)
{
	int imgW = solution.imgW;
	int imgH = solution.imgH;

	#ifdef SHOWIMAGES
		static IplImage *curvimg = 0;
		if (!curvimg)
		{
			curvimg = cvCreateImage (cvSize(imgW, imgH), IPL_DEPTH_8U, 1);

			cvvNamedWindow ("curvimg", CV_WINDOW_AUTOSIZE);
			cvMoveWindow ("curvimg", 0, 0);
		}

		cvZero (curvimg);
	#endif


	int e_curv = 0;

	for (int y = CURVRAD; y < imgH - CURVRAD; y++)
		for (int x = CURVRAD; x < imgW - CURVRAD; x++)
		{
			SurfaceModel *sm = solution.surfacemodels[y * imgW + x];

			if (sm->which_type() == IS_SPLINE)
			{
				BSpline *spline = (BSpline*) sm;
				float curv = spline->PointCurvature (cvPoint (x,y));
				
				int pen = ComputeCurvPen (curv, CurvPen);
				//int pen = (int) (curv * (float) CurvPen + 0.5f);

				//printf ("%.5f => %d\n", curv, pen);

				e_curv += pen;

				#ifdef SHOWIMAGES
					curvimg->imageData[y * curvimg->widthStep + x] = (char) min (pen, 255);
				#endif
			}
		}

	#ifdef SHOWIMAGES
		cvShowImage ("curvimg", curvimg);
		cvWaitKey(50);
	#endif

	return e_curv;
}