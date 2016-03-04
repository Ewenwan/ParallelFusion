#ifndef SURFACE_MODELS_H
#define SURFACE_MODELS_H

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include <vector>
#include <algorithm>
using namespace std;

#define INVALID -1.f

#define IS_CONSTANT_SURFACE 1
#define IS_PLANE 2
#define IS_SPLINE 3

class SurfaceModel
{
public:
	virtual float PointDisp (CvPoint point, float scale) = 0;

	virtual SurfaceModel *Clone() = 0;

	virtual vector<float> SupportVals (CvPoint point, int winSize, float scale) = 0;

	virtual int is_valid () = 0;

	virtual int which_type () = 0;

	virtual int splineorder () = 0;

	virtual void Save (FILE *fp) = 0;

	int segid;
	int proposalid;
};

class Plane : public SurfaceModel
{
public:
	float A;
	float B;
	float C;

	float PointDisp (CvPoint point, float scale);

	SurfaceModel *Clone();

	vector<float> SupportVals (CvPoint point, int winSize, float scale);

	int is_valid ();

	void Save (FILE *fp);

	int which_type ();

	int splineorder () {return 0;}
};

class BSpline : public SurfaceModel
{
public:
	CvPoint LU;

	int imgW;
	int imgH;

	int subimg_rows;
	int subimg_cols;

	int subimgW;
	int subimgH;

	BSpline () {}
	BSpline (CvPoint _LU, int _imgW, int _imgH, int _subimg_rows, int _subimg_cols);

	vector<float> knotpoints;

	vector<float> curvknotpoints_hor;
	vector<float> curvknotpoints_ver;

	int GetCurvature (float scale);

	float PointDisp (CvPoint point, float scale);

	float PointCurvature (CvPoint point);

	void FitSpline (IplImage *data, float scale);

	void FitSplineRobust (IplImage *valid_disp, IplImage *filled_disp, float scale);

	void ComputeCurvKnotPoints ();

	//void PlotSurface();

	void PlotSubImageSpline (float scale, IplImage *plot);

	int absdif (IplImage *valid, IplImage *fitted, IplImage *errorimg);

	SurfaceModel *Clone();

	vector<float> SupportVals (CvPoint point, int winSize, float scale) {vector<float> dummy; return dummy;}

	int is_valid ();

	void Save (FILE *fp) {}

	int which_type () {return IS_SPLINE;}

	int splineorder () {return subimg_rows;}

private:

	void getKnotsUV (CvPoint point, float &u, float &v, vector<float> &curknots);
};

struct lessthanSurfaceModel
{
  bool operator()(const SurfaceModel* s1, const SurfaceModel* s2) const
  {
	  if (s1->proposalid < s2->proposalid)
		  return 1;
	  else if (s1->proposalid > s2->proposalid)
		  return 0;
	  else
	  // proposalid is the same
	  {
		  if (s1->segid < s2->segid)
			  return 1;
		  else
			  return 0;
	  }
  }
};

#endif