#include "fusion.h"

#define SAVEIMAGES

void get_prefix (char *disp_fn, char *prefix)
{
	// search the .
	int i;
	for (i = strlen(disp_fn) - 1; i >= 0; i--)
	{
		if (disp_fn[i] == '.')
			break;
	}

	for (int j = 0; j < i; j++)
		prefix[j] = disp_fn[j];

	prefix[i] = '\0';
}

void get_fn (char *caption, char *disp_fn, char *fn)
{
	char prefix[2048];
	get_prefix (disp_fn, prefix);
	sprintf (fn, "%s_%s.png", prefix, caption);
}

void SaveImage (char *caption, char *disp_fn, IplImage *img)
{
	char fn[2048];

	get_fn (caption, disp_fn, fn);
	cvSaveImage (fn, img);
}

void RoundDisp (IplImage *fusedimg)
{
	for (int y = 0; y < fusedimg->height; y++)
		for (int x = 0; x < fusedimg->width; x++)
		{
			float d = (float) (uchar) fusedimg->imageData [y * fusedimg->widthStep + x];
			d /= 16.f;

			int int_d = (int) (d + 0.5f);
			int_d *= 16;

			fusedimg->imageData [y * fusedimg->widthStep + x] = (char) int_d;
		}
}

void ShowImages (QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, Proposal &fused, int OccPen, int HOPen, char *disp_fn)
{
	int imgW = proposal1.imgW;
	int imgH = proposal1.imgH;

	float scale = proposal1.scale;

	#ifdef SHOWIMAGES
		static int iteration = 0;
		iteration++;
		///////////////////////////////////////
		//		Arrange images
		if (iteration == 1)
		{
			int xspace = imgW + 20;
			int yspace = imgH + 40;

			cvvNamedWindow ("heightlines", CV_WINDOW_AUTOSIZE);
			cvMoveWindow ("heightlines", 0, 0);
			cvvNamedWindow ("orders", CV_WINDOW_AUTOSIZE);
			cvMoveWindow ("orders", 0, yspace);
			cvvNamedWindow ("prop1", CV_WINDOW_AUTOSIZE);
			cvMoveWindow ("prop1", 0, 0);
			cvvNamedWindow ("prop2", CV_WINDOW_AUTOSIZE);
			cvMoveWindow ("prop2", xspace, 0);
			cvvNamedWindow ("fused", CV_WINDOW_AUTOSIZE);
			cvMoveWindow ("fused", 2 * xspace, 0);

			cvvNamedWindow ("prop1ids", CV_WINDOW_AUTOSIZE);
			cvMoveWindow ("prop1ids", 0, yspace);
			cvvNamedWindow ("prop2ids", CV_WINDOW_AUTOSIZE);
			cvMoveWindow ("prop2ids", xspace, yspace);
			cvvNamedWindow ("fusedids", CV_WINDOW_AUTOSIZE);
			cvMoveWindow ("fusedids", 2 * xspace, yspace);
		}
	#endif

	///////////////////////////////////////////////////
	//					disparity
	IplImage *fusedimg = fused.plot_proposal(scale);

	if (proposal1.scale == 16.f)
		RoundDisp (fusedimg);
	

	IplImage *heightlines = fused.plot_height_lines (fusedimg);
	IplImage *orders = fused.plot_spline_order();

	#ifdef SHOWIMAGES
		IplImage *prop1img = proposal1.plot_proposal(scale);
		IplImage *prop2img = proposal2.plot_proposal(scale);

		cvShowImage ("prop1", prop1img);
		cvShowImage ("prop2", prop2img);
		cvShowImage ("fused", fusedimg);
		cvShowImage ("heightlines", heightlines);
		cvShowImage ("orders", orders);

		cvReleaseImage (&prop1img);
		cvReleaseImage (&prop2img);
	#endif

	SaveImage ("fused", disp_fn, fusedimg);
	SaveImage ("heightlines", disp_fn, heightlines);
	SaveImage ("orders", disp_fn, orders);

	cvReleaseImage (&fusedimg);
	cvReleaseImage (&heightlines);
	cvReleaseImage (&orders);

	///////////////////////////////////////////////////
	//					ids
	IplImage *fusedids = fused.plot_surface_ids();
	SaveImage ("fusedids", disp_fn, fusedids);
	#ifdef SHOWIMAGES
		IplImage *prop1ids = proposal1.plot_surface_ids();
		IplImage *prop2ids = proposal2.plot_surface_ids();


		cvShowImage ("prop1ids", prop1ids);
		cvShowImage ("prop2ids", prop2ids);
		cvShowImage ("fusedids", fusedids);

		cvReleaseImage (&prop1ids);
		cvReleaseImage (&prop2ids);
	#endif
	cvReleaseImage (&fusedids);

	if (OccPen > 0)
	{
		char fn[2048];
		get_fn ("occlusions", disp_fn, fn);
		ShowOcclusionGraph (q, &proposal1, &proposal2, scale, fn);
	}

	#ifdef SHOWIMAGES
		cvWaitKey(100);
	#endif
}

void showUnlabeled (QPBO<REAL>* q, IplImage *left)
{
	#ifndef SHOWIMAGES
		return;
	#endif

	IplImage *unlabelled = cvCloneImage (left);

	int imgW = left->width;
	int imgH = left->height;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			int x_i = q->GetLabel(y * imgW + x);
			
			if (x_i == -1)
			{
				for (int c = 0; c < 3; c++)
					unlabelled->imageData[y * unlabelled->widthStep + 3 * x + c] = (uchar) 0;
			}
		}

	cvvNamedWindow ("unlabelled", CV_WINDOW_AUTOSIZE);
	//cvMoveWindow ("unlabelled", 5 * imgW, 0);
	cvShowImage ("unlabelled", unlabelled);

	cvReleaseImage(&unlabelled);
}

int numUnlabeled (QPBO<REAL>* q, int imgW, int imgH)
{
	int sumUnlabeled = 0;

	// overall nodes
	for (int i = 0; i < q->GetNodeNum(); i++)
	{
		if (q->GetLabel(i) == -1)
		{
			sumUnlabeled++;
		}
	}

	printf ("%d / %d overall unlabeled nodes (%.1f\%)\n", 
			sumUnlabeled, q->GetNodeNum(), (float) sumUnlabeled / (float) q->GetNodeNum() * 100.f);

	// disparity nodes
	sumUnlabeled = 0;

	for (int i = 0; i < imgW * imgH; i++)
	{
		if (q->GetLabel(i) == -1)
		{
			sumUnlabeled++;
		}
	}

	printf ("%d / %d unlabeled disparity nodes (%.1f\%)\n", 
			sumUnlabeled, imgW * imgH, (float) sumUnlabeled / (float) (imgW * imgH) * 100.f);

	return sumUnlabeled;
}

float ComputeCurvature (CvPoint p, SurfaceModel *sm)
{
	float c = 0.f;

	float dl = sm->PointDisp (cvPoint (p.x - CURVRAD, p.y), 0.f);
	float d  = sm->PointDisp (cvPoint (p.x			, p.y), 0.f);
	float dr = sm->PointDisp (cvPoint (p.x + CURVRAD, p.y), 0.f);

	c += fabs (dl - 2.f * d + dr);

	float du = sm->PointDisp (cvPoint (p.x, p.y - CURVRAD), 0.f);
	float dd = sm->PointDisp (cvPoint (p.x, p.y + CURVRAD), 0.f);

	c += fabs (du - 2.f * d + dd);

	return c;
}

#define OUTOFGRID 999.f

inline int isInsideImage (CvPoint p, int imgW, int imgH)
{
	if (p.x < 0 || p.x >= imgW || p.y < 0 || p.y >= imgH)
		return 0;

	return 1;
}

float ComputeCurvatureSurfaceBorders (CvPoint p1, CvPoint p2, SurfaceModel *sm1, SurfaceModel *sm2, int imgW, int imgH)
{
	CvPoint p, pl, pr;
	float d, dl, dr;
	float curv = 0.f;

	CvPoint direction;
	int offset_x = (p2.x - p1.x) * CURVRAD;
	int offset_y = (p2.y - p1.y) * CURVRAD;

	p = pl = pr = p1;
	pl.x -= offset_x; pl.y -= offset_y;
	pr.x += offset_x; pr.y += offset_y;
	
	if (!isInsideImage (p, imgW, imgH) || !isInsideImage (pl, imgW, imgH) || !isInsideImage (pr, imgW, imgH)) 
		return OUTOFGRID;

	dl = sm1->PointDisp (pl, 0.f);
	d  = sm1->PointDisp (p,  0.f);
	dr = sm2->PointDisp (pr, 0.f);

	curv += fabs (dl - 2.f * d + dr);
		
	return curv;
}