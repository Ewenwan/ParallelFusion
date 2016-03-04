#include "fusion.h"

#define EIGHT_CONNECTIVITY

using namespace kolmogorov::qpbo;

int same_surface_label (SurfaceModel *s1, SurfaceModel *s2)
{
	if (s1->proposalid == s2->proposalid && s1->segid == s2->segid)
		return 1;
	else
		return 0;
}

// checks if two pixels lie on the same surface and adjust a penalty accordingly
int SmoothCosts_SurfaceLabels (SurfaceModel *s1, SurfaceModel *s2, CvPoint p1, CvPoint p2, int SmoothPen, float scale, int imgW, int imgH)
{
	int penalty = 0;

	int samesurface = same_surface_label (s1, s2);

	penalty += 1.75 * SmoothPen * !samesurface;

	if (!samesurface)
	{
		float curv = ComputeCurvatureSurfaceBorders (p1, p2, s1, s2, imgW, imgH);

		if (curv > 0.25)
			penalty += SmoothPen * 0.25;
	}

	return penalty;
}

// pairwise smoothness term (eq (5))
void AddBinaries_SurfaceLabels (QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, int SmoothPen)
{
	int imgW = proposal1.imgW;
	int imgH = proposal1.imgH;	
	float scale = proposal1.scale;

	vector<CvPoint> neighbours;
	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			SurfaceModel *cur_s1 = proposal1.surfacemodels[y * imgW + x];
			SurfaceModel *cur_s2 = proposal2.surfacemodels[y * imgW + x];

			neighbours.clear();
			
			neighbours.push_back (cvPoint (x+1,y  ));
			neighbours.push_back (cvPoint (x  ,y+1));

			#ifdef EIGHT_CONNECTIVITY
				neighbours.push_back (cvPoint (x+1,y+1));
				neighbours.push_back (cvPoint (x-1,y+1));
			#endif

			CvPoint cur_point = cvPoint (x,y);

			// for each pair of neighbouring pixels
			for (int n = 0; n < neighbours.size(); n++)
			{
				CvPoint nb = neighbours[n];

				if (nb.x < 0 || nb.x >= imgW || nb.y < 0 || nb.y >= imgH)
					continue;

				SurfaceModel *nb_s1 = proposal1.surfacemodels[nb.y * imgW + nb.x];
				SurfaceModel *nb_s2 = proposal2.surfacemodels[nb.y * imgW + nb.x];

				int E00 = SmoothCosts_SurfaceLabels (cur_s1, nb_s1, cur_point, nb, SmoothPen, scale, imgW, imgH);
				int E01 = SmoothCosts_SurfaceLabels (cur_s1, nb_s2, cur_point, nb, SmoothPen, scale, imgW, imgH);
				int E10 = SmoothCosts_SurfaceLabels (cur_s2, nb_s1, cur_point, nb, SmoothPen, scale, imgW, imgH);
				int E11 = SmoothCosts_SurfaceLabels (cur_s2, nb_s2, cur_point, nb, SmoothPen, scale, imgW, imgH);

				q->AddPairwiseTerm(y * imgW + x, nb.y * imgW + nb.x,
									E00,  // E00
									E01,  // E01
									E10,  // E10
									E11); // E11

			}
		}
}

// evaluates smoothness term (eq (5))
int EvaluateBinaries_SurfaceLabels (Proposal &solution, int SmoothPen)
{
	int imgW = solution.imgW;
	int imgH = solution.imgH;

	float scale = solution.scale;

	#ifdef SHOWIMAGES
		static IplImage *smoothpenimg = 0;
		if (!smoothpenimg)
		{
			smoothpenimg = cvCreateImage (cvSize(imgW, imgH), IPL_DEPTH_8U, 3);

			cvvNamedWindow ("smoothpen", CV_WINDOW_AUTOSIZE);
			cvMoveWindow ("smoothpen", 0, 2 * imgH + 80);
		}

		cvZero (smoothpenimg);
	#endif

	int e_smooth = 0;

	vector<CvPoint> neighbours;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			SurfaceModel *cur_s = solution.surfacemodels[y * imgW + x];

			neighbours.clear();
			neighbours.push_back (cvPoint (x+1,y  ));
			neighbours.push_back (cvPoint (x  ,y+1));

			#ifdef EIGHT_CONNECTIVITY
				neighbours.push_back (cvPoint (x+1,y+1));
				neighbours.push_back (cvPoint (x-1,y+1));
			#endif

			CvPoint cur_point = cvPoint (x,y);

			int gets_small_pen = 0;
			int gets_large_pen = 0;

			for (int n = 0; n < neighbours.size(); n++)
			{
				CvPoint nb = neighbours[n];

				if (nb.x < 0 || nb.x >= imgW || nb.y < 0 || nb.y >= imgH)
					continue;

				SurfaceModel *nb_s = solution.surfacemodels[nb.y * imgW + nb.x];

				int c = SmoothCosts_SurfaceLabels (cur_s, nb_s, cur_point, nb, SmoothPen, scale, imgW, imgH);
				e_smooth += c;

				if (c == SmoothPen)
					gets_small_pen = 1;
				if (c == 2 * SmoothPen)
					gets_large_pen = 1;
			}

			#ifdef SHOWIMAGES
				if (gets_small_pen)
					smoothpenimg->imageData[y * smoothpenimg->widthStep + 3 * x + 2] = (uchar) 122;
				if (gets_large_pen)
					for (int i = 0; i < 3; i++)
						smoothpenimg->imageData[y * smoothpenimg->widthStep + 3 * x + i] = (uchar) 255;
			#endif
		}

	#ifdef SHOWIMAGES
		cvShowImage ("smoothpen", smoothpenimg);
	#endif

	return e_smooth;
}