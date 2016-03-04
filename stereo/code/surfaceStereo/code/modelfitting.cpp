#include "modelfitting.h"

#include <opencv2/opencv.hpp>
/////////////////////////////////////////////////////////////
//						Planes
/////////////////////////////////////////////////////////////

Plane FitPlane(vector<CvPoint3D32f> Points, FILE *fp = 0)
// least squared error fitting
{
	double x2 = 0.0;
	double y2 = 0.0;
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;
	double xy = 0.0;
	double xz = 0.0;
	double yz = 0.0;
	Plane plane;

	if (Points.size() >= 3)
	{
		for (unsigned int i = 0; i < Points.size(); i++)
		{
			x2 += Points[i].x * Points[i].x;
			y2 += Points[i].y * Points[i].y;
			x += Points[i].x;
			y += Points[i].y;
			z += Points[i].z;
			xy += Points[i].x * Points[i].y;
			xz += Points[i].x * Points[i].z;
			yz += Points[i].y * Points[i].z;
		}

		CvMat A;
		double A_Data[9];
		cvInitMatHeader (&A, 3, 3, CV_64FC1, A_Data);

		A_Data[0] = x2;		A_Data[1] = xy;		A_Data[2] = x;
		A_Data[3] = xy;		A_Data[4] = y2;		A_Data[5] = y;
		A_Data[6] = x;		A_Data[7] = y;		A_Data[8] = (double) Points.size();

		CvMat Inv;
		double Inv_Data[9];
		cvInitMatHeader (&Inv, 3, 3, CV_64FC1, Inv_Data);

		cv::Mat A2(3,3,CV_64FC1);
		double* pA2 = (double*)A2.data;
		for(int i=0; i<9; ++i)
			pA2[i] = A_Data[i];
		A2.inv();
		for(int i=0; i<9; ++i)
			Inv_Data[i] = pA2[i];

		CvMat t;
		double t_Data[9];
		cvInitMatHeader (&t, 3, 1, CV_64FC1, t_Data);

		t_Data[0] = xz;
		t_Data[1] = yz;
		t_Data[2] = z;

		CvMat Result;
		double Result_Data[3];
		cvInitMatHeader (&Result, 3, 1, CV_64FC1, Result_Data);

		cvMatMulAdd (&Inv, &t, 0, &Result);		

		plane.A = (float) Result_Data[0];
		plane.B = (float) Result_Data[1];
		plane.C = (float) Result_Data[2];

		if (fp)
		{
			fprintf (fp, "P = [\n");
			for (unsigned int i = 0; i < Points.size(); i++) 
				fprintf (fp, "%f\t%f\t%f\n", Points[i].x, Points[i].y, Points[i].z);
			fprintf (fp, "]\n");

			//PrintMartix (A_Data, 3, 3, "A");
			//PrintMartix (Inv_Data, 3, 3, "Ainv");
			//PrintMartix (t_Data, 1, 3, "t");
			fprintf (fp, "plane = [%f,\t%f\t%f]\n", plane.A, plane.B, plane.C);

			fprintf (fp, "hold on\n");
			fprintf (fp, "grid on\n");
			fprintf (fp, "plot3(P(:,1), P(:,2), P(:,3), 'r>')\n");
			fprintf (fp, "plot3 (P(:,1), P(:,2), P(:,1) * plane(1) + P(:,2) * plane(2) + plane(3))\n");
		}

		if (plane.A == 1.0)
		{
			plane.A = INVALID;
			plane.B = INVALID;
			plane.C = INVALID;
		}

	}
	else
	{
		plane.A = INVALID;
		plane.B = INVALID;
		plane.C = INVALID;
	}

	return plane;
}

Plane FitConstantPlane(vector<CvPoint3D32f> Points)
{
	printf ("constant plane\n");

	// compute max disp value
	float maxd = -1.f;

	for (unsigned int i = 0; i < Points.size(); i++)
		if (Points[i].z > maxd)
			maxd = Points[i].z;

	vector<int> histogram;
	histogram.resize ((int) maxd + 2);

	for (unsigned int i = 0; i < Points.size(); i++)
	{
		//printf ("%d\n", (int) (Points[i].z + 0.5));

		histogram[(int) (Points[i].z + 0.5)]++;
	}

	float maxbin = -1;
	float bestd = -1;
	for (unsigned int i = 0; i < histogram.size(); i++)
		if (histogram[i] > maxbin)
		{
			maxbin = histogram[i];
			bestd = i;
		}

	Plane plane;
	plane.A = plane.B = 0.f;
	plane.C = bestd;

	return plane;
}

void FitConstantPlaneToSegment (Segment &segment, int isReference, IplImage *disp, float scale)
{
	vector<CvPoint3D32f> points;

	for (unsigned int i = 0; i < segment.segpoints.size(); i++)
	{
		CvPoint3D32f point;
		int x = segment.segpoints[i].x;
		int y = segment.segpoints[i].y;

		float d = ((float) (uchar) disp->imageData [y * disp->widthStep + x]) / scale;

		point.x = (float) x;
		point.y = (float) y;
		point.z = (float) d;

		points.push_back (point);
	}

	*((Plane*) segment.surfacemodel) = FitConstantPlane(points);
	segment.surfacemodel->segid = segment.segid;

/*	if (!isReference)
	{
		segment.plane.A *= -1.f;
		segment.plane.B *= -1.f;
		segment.plane.C *= -1.f;
	}*/
}

IplImage *PlotDisparityPlanes (vector<Segment> segments, int imgW, int imgH, float scale)
{
	IplImage *disp = cvCreateImage (cvSize (imgW, imgH), IPL_DEPTH_8U, 1);
	cvZero (disp);

	for (unsigned int i = 0; i < segments.size(); i++)
	{
		for (unsigned int j = 0; j < segments[i].segpoints.size(); j++)
		{
			int x = segments[i].segpoints[j].x;
			int y = segments[i].segpoints[j].y;

			Plane plane = *((Plane*) (segments[i].surfacemodel));

			float d = plane.A * (float) x + plane.B * (float) y + plane.C;
			d = min (d * scale, 255.f);
			
			disp->imageData[y * disp->widthStep + x] = (uchar) (int) d;
		}
	}

	return disp;
}

int is_outlier (CvPoint coord, Plane plane, float real_disp, float threshold)
{
	if (plane.A == INVALID)
		return 0;

	float d = plane.A * coord.x + plane.B * coord.y + plane.C;

	if (abs (d - real_disp) > threshold)
		return 1;
	else
		return 0;
}

int isEqualPlane (Plane plane1, Plane plane2)
{
	#define PLANETHRESH 0.00000001

	if (abs(plane1.A - plane2.A) > PLANETHRESH)
		return 0;
	if (abs(plane1.B - plane2.B) > PLANETHRESH)
		return 0;
	if (abs(plane1.C - plane2.C) > PLANETHRESH)
		return 0;

	return 1;
}

float get_disparity (Plane plane, CvPoint point)
{
	return plane.A * (float) point.x + plane.B * (float) point.y + plane.C;
}

float XSteepness (Plane plane)
{
	CvPoint x1 = cvPoint (0,0);
	float d1 = get_disparity (plane, x1);

	CvPoint x2 = cvPoint (100,0);
	float d2 = get_disparity (plane, x2);

	return abs (d1 - d2);
}

void FitPlaneToSegment_ROBUST (Segment &segment, int isReference, IplImage *disp, float scale, float plane_threshold)
{
	Plane *plane = (Plane*) malloc (sizeof (Plane));
	plane->A = INVALID;

	for (int iteration = 0; iteration < 10; iteration++)
	{
		vector<CvPoint3D32f> points;

		for (unsigned int i = 0; i < segment.segpoints.size(); i++)
		{
			CvPoint3D32f point;
			int x = segment.segpoints[i].x;
			int y = segment.segpoints[i].y;

			float d = ((float) (uchar) disp->imageData [y * disp->widthStep + x]) / scale;

			if (!is_outlier (cvPoint(x,y), *plane, d, plane_threshold))
			{
				point.x = (float) x;
				point.y = (float) y;
				point.z = (float) d;

				points.push_back (point);
			}
		}

		Plane tmp = FitPlane(points);
		plane->A = tmp.A;
		plane->B = tmp.B;
		plane->C = tmp.C;
	}

	segment.surfacemodel = (SurfaceModel*) plane;
	segment.surfacemodel->segid = segment.segid;
}

/////////////////////////////////////////////////////////////
//						Splines
/////////////////////////////////////////////////////////////

void GetSubImageDimensions (Segment &segment, CvPoint &LU, int &simgW, int &simgH)
{
	int x_min = 999999;
	int x_max = 0;
	int y_min = 999999;
	int y_max = 0;

	for (int i = 0; i < segment.segpoints.size(); i++)
	{
		int x = segment.segpoints[i].x;
		int y = segment.segpoints[i].y;

		if (x < x_min)
			x_min = x;
		if (x > x_max)
			x_max = x;

		if (y < y_min)
			y_min = y;
		if (y > y_max)
			y_max = y;
	}

	LU = cvPoint (x_min, y_min);
	simgW = x_max - x_min + 1;
	simgH = y_max - y_min + 1;
}

void FillImageWithPlane (Segment &segment, IplImage *disp, float scale, CvPoint LU, IplImage *dst)
{
	// get plane first
	FitPlaneToSegment_ROBUST (segment, 1, disp, scale, 1.0);

	Plane *plane = new Plane;
	*plane = *(Plane*) (segment.surfacemodel);

	cvZero(dst);
	for (int y = 0; y < dst->height; y++)
		for (int x = 0; x < dst->width; x++)
		{
			float d = plane->PointDisp (cvPoint (LU.x + x, LU.y + y), scale) * scale;
			dst->imageData [y * dst->widthStep + x] = (uchar) (int) d + 0.5f;
		}

	delete plane;
}

void CopyDispToSegmentPoints (Segment &segment, IplImage *disp, CvPoint LU, IplImage *dst)
{
	for (int i = 0; i < segment.segpoints.size(); i++)
	{
		CvPoint p = segment.segpoints[i];

		int d = (int) (uchar) disp->imageData[p.y * disp->widthStep + p.x];

		p.x -= LU.x;
		p.y -= LU.y;

		dst->imageData [p.y * dst->widthStep + p.x] = (uchar) d;
	}
}

void PropagateDispToUnassignedRegions (IplImage *disp)
{
	int imgW = disp->width;
	int imgH = disp->height;

	for (int y = 0; y < imgH; y++)
	{
		int filldisp = -1;
		for (int x = 0; x < imgW; x++)
		{
			int curd = (int) (uchar) disp->imageData [y * disp->widthStep + x];

			if (curd == 0 && filldisp != -1)
				disp->imageData [y * disp->widthStep + x]= (uchar) filldisp;
			if (curd != 0)
				filldisp = curd;
		}

		filldisp = -1;
		for (int x = imgW - 1; x >= 0; x--)
		{
			int curd = (int) (uchar) disp->imageData [y * disp->widthStep + x];

			if (curd == 0 && filldisp != -1)
				disp->imageData [y * disp->widthStep + x]= (uchar) filldisp;
			if (curd != 0)
				filldisp = curd;
		}
	}
}

void FitSplineToSegment (Segment &segment, IplImage *disp, float scale, int splinenumber)
{
	CvPoint LU;
	int simgW, simgH;
	GetSubImageDimensions (segment, LU, simgW, simgH);

	// generate the image that we use for fitting
	IplImage *valid_disp = cvCreateImage (cvSize (simgW, simgH), IPL_DEPTH_8U, 1);
	cvZero(valid_disp);

	CopyDispToSegmentPoints (segment, disp, LU, valid_disp);

	//#define SHOW_FITTING_STEPS
	#ifdef SHOW_FITTING_STEPS
		int minsegsize = 5000;
		if (segment.segpoints.size() > minsegsize)
		{
			cvvNamedWindow ("valid_disp", CV_WINDOW_AUTOSIZE);
			cvMoveWindow ("valid_disp", 0, 0);
			cvShowImage ("valid_disp", valid_disp);
		}
	#endif

	IplImage *filled_disp = cvCreateImage (cvSize (simgW, simgH), IPL_DEPTH_8U, 1);
	cvZero (filled_disp);
	FillImageWithPlane (segment, disp, scale, LU, filled_disp);
	//CopyDispToSegmentPoints (segment, disp, LU, filled_disp);
	//PropagateDispToUnassignedRegions (filled_disp);

	#ifdef SHOW_FITTING_STEPS
		if (segment.segpoints.size() > minsegsize)
		{
			cvvNamedWindow ("filled_disp", CV_WINDOW_AUTOSIZE);
			cvMoveWindow ("filled_disp", disp->width, 0);
			cvShowImage ("filled_disp", filled_disp);
		}
	#endif

	int cols, rows;
	if (simgW < 10 || simgH < 10)
		cols = rows = 1;
	else
		cols = rows = splinenumber;

	BSpline *spline = new BSpline (LU, simgW, simgH, rows, cols);

	//spline->FitSpline (filled_disp, scale);
	spline->FitSplineRobust (valid_disp, filled_disp, scale);

	#ifdef SHOW_FITTING_STEPS
		if (segment.segpoints.size() > minsegsize)
		{
			spline->FitSplineRobust (valid_disp, filled_disp, scale);

			//spline->FitSplineRobust (valid_disp, filled_disp, scale);

			//IplImage *showdisp = spline->PlotSubImageSpline (scale);
			//spline->absdif (valid_disp, showdisp);

			


			//cvvNamedWindow ("spline", CV_WINDOW_AUTOSIZE);
			//cvMoveWindow ("spline", 2 * disp->width, 0);
			//cvShowImage ("spline", showdisp);
			//cvWaitKey(0);
		}

		//cvReleaseImage (&showdisp);
	#endif

	cvReleaseImage (&valid_disp);
	cvReleaseImage (&filled_disp);

	segment.surfacemodel = (SurfaceModel *) spline;
}