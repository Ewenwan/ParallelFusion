#include "surfacemodels.h"

BSpline::BSpline (CvPoint _LU, int _imgW, int _imgH, int _subimg_rows, int _subimg_cols)
{
	LU = _LU;
	imgW = _imgW;
	imgH = _imgH;
	subimg_rows = _subimg_rows;
	subimg_cols = _subimg_cols;
}

void BlendingFunctions (float u, float *res)
{
	// implements open uniform B-Spline Curve from Hearn and Baker pp 448
	
	// scale value (in the book control points are between 0 and 3)
	u *= 3.f;

	for (int i = 0; i < 5; i++)
		res[i] = 0.f;

	// B_0,3
	if (u < 0.f)
		res[0] = 1.f;
	if (u <= 1.f)
	{
		res[0] = 1.f - u;
		res[0] *= res[0];
	}

	// B_1,3
	if (0.f <= u && u < 1.f)
	{
		res[1] = 0.5f * u * (4.f - 3.f * u);
	}
	else if (1.f <= u && u < 2.f)
	{
		res[1] = 0.5f * (2.f - u) * (2.f - u);
	}

	// B_2,3
	if (0.f <= u && u < 1.f)
	{
		res[2] = 0.5f * u * u;
	}
	else if (1.f <= u && u < 2.f)
	{
		res[2] = 0.5f * u * (2.f - u) + 0.5f * (u - 1.f) * (3.f - u);
	}
	else if (2.f <= u && u < 3.f)
	{
		res[2] = 0.5f * (3.f - u) * (3.f - u);
	}

	// B_3,3
	if (1.f <= u && u < 2.f)
	{
		res[3] = 0.5f * (u - 1.f) * (u - 1.f);
	}
	else if (2.f <= u && u < 3.f)
	{
		res[3] = 0.5f * (3.f - u) * (3.f * u - 5.f);
	}

	// B_4,3
	if (2.f <= u && u < 3.f)
	{
		res[4] = (u - 2.f) * (u - 2.f);
	}
	if (3.f <= u)
	{
		res[4] = 1.f;
	}
}

void BSpline::getKnotsUV (CvPoint point, float &u, float &v, vector<float> &curknots)
{
	int x = point.x / (subimgW);
	int y = point.y / (subimgH);

	if (x >= subimg_cols)
		x = subimg_cols - 1;
	if (y >= subimg_rows)
		y = subimg_rows - 1;


	curknots.clear();
	for (int i = 0; i < 25; i++)
		curknots.push_back(knotpoints[y * subimg_cols * 25 + x * 25 + i]);

	u = (float) (point.x % subimgW);
	u /= (float) (subimgW - 1);

	v = (float) (point.y % subimgH);
	v /= (float) (subimgH - 1);

	/*if (u == 1.f)	
		printf ("u %.3f v %.3f\n", u, v);*/
}

float BSpline::PointDisp (CvPoint point, float scale)
{
	point.x -= LU.x;
	point.y -= LU.y;

	// small enlargement allowed

	int tolerance = 25;

	if (point.x < 0)
	{
		if (point.x > -tolerance)
			point.x = 0;
		else
			return -1.f;
	}
	if (point.x >= imgW)
	{
		if (point.x < imgW + tolerance)
			point.x = imgW - 1;
		else
			return -1.f;
	}

	if (point.y < 0)
	{
		if (point.y > -tolerance)
			point.y = 0;
		else
			return -1.f;
	}
	if (point.y >= imgH)
	{
		if (point.y < imgH + tolerance)
			point.y = imgH - 1;
		else
			return -1.f;
	}


	float d;

	float u = 0.f;
	float v = 0.f;
	static vector<float> curknots;
	curknots.clear();

	getKnotsUV (point, u, v, curknots);

/*	float u = (float) point.x / (float) (imgW - 1);
	float v = (float) point.y / (float) (imgH - 1);*/

	float u_blends[5];
	float v_blends[5];
	
	BlendingFunctions (u, u_blends);
	BlendingFunctions (v, v_blends);

	d = 0.f;

	for (int y = 0; y < 5; y++)
		for (int x = 0; x < 5; x++)
		{
			d += curknots[y * 5 + x] * u_blends[x] * v_blends[y];
		}

	//int d_int = (int) (d * scale + 0.5);

	//d = (float) d_int / scale;

	return d;	
}

void BSpline::FitSpline (IplImage *data, float scale)
{
	knotpoints.clear();

	subimgW = imgW / subimg_cols;
	subimgH = imgH / subimg_rows;

	for (int sy = 0; sy < subimg_rows; sy++)
		for (int sx = 0; sx < subimg_cols; sx++)
			for (int y = 0; y < 5; y++)
				for (int x = 0; x < 5; x++)
				{
					int ux = sx * subimgW + x * (subimgW - 1) / 4;
					int uy = sy * subimgH + y * (subimgH - 1) / 4;

					//printf ("ux %d uy %d\n", ux, uy);

					float d = (float) (uchar) data->imageData[uy * data->widthStep + ux];
					d /= scale;

					knotpoints.push_back (d);
				}
}

void BSpline::FitSplineRobust (IplImage *valid_disp, IplImage *filled_disp, float scale)
{
/*	static vector<float> *candidates = 0;
	if (!candidates)
		candidates = new vector<float>;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			float d = (float) (uchar) valid_disp->imageData[y * valid_disp->widthStep + x];
			d /= scale;
	
			if (d == 0.f)
				continue;
			else
				candidates->push_back(d);
		}*/

	IplImage *splineimg = cvCreateImage (cvSize (imgW, imgH), IPL_DEPTH_8U, 1);
	IplImage *errorimg = cvCreateImage (cvSize (imgW, imgH), IPL_DEPTH_8U, 3);

	FitSpline (filled_disp, scale);
	PlotSubImageSpline (scale, splineimg);
	int best_error = absdif(valid_disp, splineimg, errorimg);
	static vector<float> bestsolution;
	bestsolution = knotpoints;

/*	cvvNamedWindow ("spline", CV_WINDOW_AUTOSIZE);
	cvMoveWindow ("spline", 0, 400);
	cvShowImage ("spline", splineimg);
	cvWaitKey(0);*/


	int numchanges = 1;//knotpoints.size() / 5;

	int iterations = 0;

	while (1)
	{
		if (iterations > knotpoints.size() * 3)
			break;

		iterations++;

		int knot_idx = iterations % knotpoints.size();
		//int cand_idx = rand() % candidates.size();

		float curd = knotpoints[knot_idx];

		int bestup_costs, bestdown_costs;
		float bestup_d, bestdown_d;
//		bestup_costs = bestdown_costs = best_error;
//		bestup_d = bestdown_d = curd;

		for (int i = 0; i < 2; i++)
		{
			int c = best_error; 
			float d = curd;
			knotpoints[knot_idx] = curd;

			int improvement = 1;
			while (improvement)
			{
				if (i == 0)
					knotpoints[knot_idx] += 0.5;
				else
					knotpoints[knot_idx] -= 0.5;

				PlotSubImageSpline (scale, splineimg);
				int error = absdif(valid_disp, splineimg, errorimg);

				if (error < c)
				{
					c = error;
					d = knotpoints[knot_idx];
				}
				else
					improvement = 0;
			}

			if (i == 0)
			{
				bestup_costs = c;
				bestup_d = d;
			}
			else
			{
				bestdown_costs = c;
				bestdown_d = d;
			}
		}

		if (bestdown_costs < bestup_costs)
		{
			best_error = bestdown_costs;
			knotpoints[knot_idx] = bestdown_d;
		}
		else
		{
			best_error = bestup_costs;
			knotpoints[knot_idx] = bestup_d;
		}

		//printf ("best error %d\n", best_error);
	}

/*	cvvNamedWindow ("spline", CV_WINDOW_AUTOSIZE);
	cvMoveWindow ("spline", 0, 400);
	cvShowImage ("spline", splineimg);

/*	cvvNamedWindow ("error", CV_WINDOW_AUTOSIZE);
	cvMoveWindow ("error", 0, 200);
	cvShowImage ("error", errorimg);*/

	//cvWaitKey(0);

	cvReleaseImage (&splineimg);
	cvReleaseImage (&errorimg);


	// precompute secondorderderivative
	ComputeCurvKnotPoints();
}

/*void BSpline::FitSplineRobust (IplImage *valid_disp, IplImage *filled_disp, float scale)
{
	printf (".");

	static vector<float> candidates;
	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			float d = (float) (uchar) valid_disp->imageData[y * valid_disp->widthStep + x];
			d /= scale;
	
			if (d == 0.f)
				continue;
			else
				candidates.push_back(d);
		}

	IplImage *splineimg = cvCreateImage (cvSize (imgW, imgH), IPL_DEPTH_8U, 1);
	IplImage *errorimg = cvCreateImage (cvSize (imgW, imgH), IPL_DEPTH_8U, 3);

	FitSpline (filled_disp, scale);
	PlotSubImageSpline (scale, splineimg);
	int best_error = absdif(valid_disp, splineimg, errorimg);
	static vector<float> bestsolution;
	bestsolution = knotpoints;

	//cvvNamedWindow ("spline", CV_WINDOW_AUTOSIZE);
	//cvMoveWindow ("spline", 0, 400);
	//cvShowImage ("spline", splineimg);
	//cvWaitKey(0);


	int numchanges = 1;//knotpoints.size() / 5;

	int iterations = 0;

	while (1)
	{
		if (iterations > 5000)
			break;

		iterations++;

		for (int i = 0; i < numchanges; i++)
		{
			int knot_idx = rand() % knotpoints.size();
			int cand_idx = rand() % candidates.size();

			knotpoints[knot_idx] = candidates[cand_idx];
		}

		PlotSubImageSpline (scale, splineimg);
		int error = absdif(valid_disp, splineimg, errorimg);

		//printf ("%d, %d\n", error, best_error);

		if (error > best_error)
			knotpoints = bestsolution;
		else
		{
			best_error = error;
			bestsolution = knotpoints;			
		}
	}

	cvvNamedWindow ("spline", CV_WINDOW_AUTOSIZE);
	cvMoveWindow ("spline", 0, 400);
	cvShowImage ("spline", splineimg);

	cvvNamedWindow ("error", CV_WINDOW_AUTOSIZE);
	cvMoveWindow ("error", 0, 200);
	cvShowImage ("error", errorimg);

	cvWaitKey(0);

	cvReleaseImage (&splineimg);
	cvReleaseImage (&errorimg);

}*/

void BSpline::PlotSubImageSpline (float scale, IplImage *plot)
{
	cvZero (plot);

	for (int y = 0; y < imgH; y+=2)
		for (int x = 0; x < imgW; x+=2)
		{
			float d = PointDisp (cvPoint (LU.x + x, LU.y + y), scale) * scale;
			plot->imageData [y * plot->widthStep + x] = (uchar) (int) d + 0.5f;
		}
}

int BSpline::absdif (IplImage *valid, IplImage *fitted, IplImage *errorimg)
{
	//cvZero (errorimg);

	int error = 0;

	for (int y = 0; y < imgH; y+= 2)
		for (int x = 0; x < imgW; x+= 2)
		{
			int d = (int) (uchar) valid->imageData[y * valid->widthStep + x];

			if (!d)
			{
				//errorimg->imageData[y * errorimg->widthStep + 3 * x + 0] = 0;
				//errorimg->imageData[y * errorimg->widthStep + 3 * x + 1] = 0;
				//errorimg->imageData[y * errorimg->widthStep + 3 * x + 2] = 255;

				continue;
			}

			int d_fitted = (int) (uchar) fitted->imageData[y * fitted->widthStep + x];
			int dif = abs (d - d_fitted);
			error += dif;

			//for (int c = 0; c < 3; c++)
			//	errorimg->imageData[y * errorimg->widthStep + 3 * x + c] = (uchar) min (dif * 20, 255);
		}

	if (!is_valid())
		error *= 2;

	return error;
}

SurfaceModel *BSpline::Clone()
{
	BSpline *spline = new BSpline(LU, imgW, imgH, subimg_rows, subimg_cols);

	spline->imgW = imgW;
	spline->imgH = imgH;

	spline->subimg_rows = subimg_rows;
	spline->subimg_cols = subimg_cols;

	spline->subimgW = subimgW;
	spline->subimgH = subimgH;

	spline->knotpoints = knotpoints;

	spline->curvknotpoints_hor = curvknotpoints_hor;
	spline->curvknotpoints_ver = curvknotpoints_ver;

	spline->proposalid = proposalid;
	spline->segid = segid;

	return (SurfaceModel*) spline;

	return 0;
}

float gradient (float d1, float d2, float spacing)
{
	return abs (d1 - d2) / spacing;
}

int BSpline::is_valid()
{
	float spacing = (float) imgW / ((float) splineorder() * 4.f);

	// a control point every 2 pixel is ok. if spacing is less invalidate surface model
	if (spacing < 2.f)
		return 0;

	float grad_threshold = 0.25f;

	// check the gradient;
	for (int y = 0; y < 4; y++)
		for (int x = 0; x < 4; x++)
		{
			float d1 = knotpoints[y * 5 + x];
			float d2 = knotpoints[y * 5 + x + 1];

			float grad = gradient (d1, d2, spacing);
			if (grad > grad_threshold)
				return 0;

			d1 = knotpoints[y * 5 + x];
			d2 = knotpoints[(y + 1) * 5 + x];

			grad = gradient (d1, d2, spacing);
			if (grad > grad_threshold)
				return 0;

		}
	return 1;
}

int BSpline::GetCurvature (float scale)
{
	float sum = 0.f;

	// horizontal
	for (int y = 0; y < imgH; y++)
		for (int x = 1; x < imgW - 1; x++)
		{
			float dl = PointDisp (cvPoint (LU.x + x - 1, LU.y + y), scale);
			float d  = PointDisp (cvPoint (LU.x + x    , LU.y + y), scale);
			float dr = PointDisp (cvPoint (LU.x + x + 1, LU.y + y), scale);

			float c = fabs (dl - 2 * d + dr);
			c *= c;

			sum += c;
		}

	// vertical
	for (int y = 1; y < imgH - 1; y++)
		for (int x = 0; x < imgW; x++)
		{
			float dl = PointDisp (cvPoint (LU.x + x, LU.y + y - 1), scale);
			float d  = PointDisp (cvPoint (LU.x + x, LU.y + y    ), scale);
			float dr = PointDisp (cvPoint (LU.x + x, LU.y + y + 1), scale);

			float c = fabs (dl - 2 * d + dr);
			c *= c;

			sum += c;
		}

	//printf ("curvature %d", (int) (sum + 0.5));

	return (int) (sum + 0.5);
}

//////////////////////////////////////////////////////////
//			Curvature									//
//////////////////////////////////////////////////////////

void FirstOrderControlPoints (vector<float> controlpoints, vector<float> &newcontrolpoints)
{
	float u[8];
	u[0] = 0.f; u[1] = 0.f; u[2] = 0.f; u[3] = 1.f; 
	u[4] = 2.f; u[5] = 3.f; u[6] = 3.f; u[7] = 3.f;

	for (int i = 0; i < 4; i++)
	{
		float Q_i;
		// degree
		int p = 2;

		Q_i = 2.f / ((float) u[i + p + 1] - (float) u[i + 1]);

		float P_i_plus_1 = controlpoints[i + 1];
		float P_i        = controlpoints[i];

		Q_i *= (P_i_plus_1 - P_i);

		//Q_i = fabs (P_i_plus_1 - P_i);

		newcontrolpoints.push_back(Q_i);
	}
}

void SecondOrderControlPoints (vector<float> controlpoints, vector<float> &newcontrolpoints)
{
	float u[6];
	u[0] = 0.f; u[1] = 0.f; u[2] = 1.f; 
	u[3] = 2.f; u[4] = 3.f; u[5] = 3.f;

	for (int i = 0; i < 3; i++)
	{
		float Q_i;
		// degree
		int p = 1;

		Q_i = 2.f / ((float) u[i + p + 1] - (float) u[i + 1]);

		float P_i_plus_1 = controlpoints[i + 1];
		float P_i        = controlpoints[i];

		Q_i *= (P_i_plus_1 - P_i);

		//Q_i = fabs (Q_i);

		newcontrolpoints.push_back(Q_i);
	}
}

void BSpline::ComputeCurvKnotPoints ()
{
	curvknotpoints_hor.resize (3 * 5);
	curvknotpoints_ver.resize (5 * 3);

	// horizontal ones
	for (int y = 0; y < 5; y++)
	{
		vector<float> controlpoints;
		vector<float> firstorder;
		vector<float> secondorder;

		for (int x = 0; x < 5; x++)
			controlpoints.push_back(knotpoints[5 * y + x]);

		FirstOrderControlPoints (controlpoints, firstorder);
		SecondOrderControlPoints (firstorder, secondorder);

		for (int x = 0; x < 3; x++)
			curvknotpoints_hor[y * 3 + x] = secondorder[x];
	}

	// vertical ones
	for (int x = 0; x < 5; x++)
	{
		vector<float> controlpoints;
		vector<float> firstorder;
		vector<float> secondorder;

		for (int y = 0; y < 5; y++)
			controlpoints.push_back(knotpoints[5 * y + x]);

		FirstOrderControlPoints (controlpoints, firstorder);
		SecondOrderControlPoints (firstorder, secondorder);

		for (int y = 0; y < 3; y++)
			curvknotpoints_ver[y * 5 + x] = secondorder[y];
	}


	PointCurvature (cvPoint(LU.x + 5, LU.y + 5));
}

void CurvatureBlendingFunctions (float u, float *res)
{
	u *= 3.f;

	for (int i = 0; i < 3; i++)
		res[i] = 0.f;

	// B_0,3
	if (u <= 1.f)
	{
		res[0] = 1.f;
		return;
	}
	if (u <= 2.f)
	{
		res[1] = 1.f;
		return;
	}

	res[2] = 1.f;
	return;
}

float BSpline::PointCurvature (CvPoint point)
{
	point.x -= LU.x;
	point.y -= LU.y;

	// small enlargement allowed

	int tolerance = 25;

	if (point.x < 0)
	{
		if (point.x > -tolerance)
			point.x = 0;
		else
			return 999.f;
	}
	if (point.x >= imgW)
	{
		if (point.x < imgW + tolerance)
			point.x = imgW - 1;
		else
			return 999.f;
	}

	if (point.y < 0)
	{
		if (point.y > -tolerance)
			point.y = 0;
		else
			return 999.f;
	}
	if (point.y >= imgH)
	{
		if (point.y < imgH + tolerance)
			point.y = imgH - 1;
		else
			return 999.f;
	}

	float u = 0.f;
	float v = 0.f;
	static vector<float> curknots;
	curknots.clear();

	getKnotsUV (point, u, v, curknots);

	float u_blends[5];
	float v_blends[5];
	
	BlendingFunctions (u, u_blends);
	BlendingFunctions (v, v_blends);

	float curv_u_blends[3];
	float curv_v_blends[3];

	CurvatureBlendingFunctions (u,curv_u_blends);
	CurvatureBlendingFunctions (v,curv_v_blends);

	float curv = 0.f;

	for (int y = 0; y < 5; y++)
		for (int x = 0; x < 3; x++)
		{
			curv += curvknotpoints_hor[y * 3 + x] * curv_u_blends[x] * v_blends[y];
		}

	for (int x = 0; x < 5; x++)
		for (int y = 0; y < 3; y++)
		{
			curv += curvknotpoints_ver[y * 5 + x] * u_blends[x] * curv_v_blends[y];
		}


	curv = fabs (curv);

	//printf ("%.3f\n", curv);

	return curv;	
}