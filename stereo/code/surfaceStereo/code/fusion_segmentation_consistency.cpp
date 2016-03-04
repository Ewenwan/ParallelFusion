#include "fusion.h"

using namespace kolmogorov::qpbo;
// this is the sparse higher order construction as described in Carsten's paper
// this here is the type 1 construction
void AddType1Construction(QPBO<REAL>* q, vector<CvPoint> points, vector<int> state, int PH, int imgW)
{
	//printf ("TYPE 1 used\n");

	int z0 = q->AddNode(1);
	int z1 = q->AddNode(1);

	q->AddUnaryTerm(z0, 0, PH);
	q->AddUnaryTerm(z1, PH, 0);
	
	q->AddPairwiseTerm(z0, z1,
						0,		// E00
						0,		// E01
						-PH,	// E10
						0);		// E11

	for (int i = 0; i < points.size(); i++)
	{
		if (state[i] == 1)
		{
			q->AddPairwiseTerm(z1, points[i].y * imgW + points[i].x,
								0,		// E00
								0,		// E01
								PH,		// E10
								0);		// E11
		}
		if (state[i] == 0)
		{
			q->AddPairwiseTerm(z0, points[i].y * imgW + points[i].x,
								0,		// E00
								PH,		// E01
								0,		// E10
								0);		// E11
		}
	}
}

// this is the PN Potts model construction of Pushmeet's P3 and beyond paper
void P3andBeyondConstruction(QPBO<REAL>* q, vector<CvPoint> *points, vector<int> *state, int sig_alpha, int sig, int sig_max, int imgW)
{
	int Ms = q->AddNode(1);
	int Mt = q->AddNode(1);

	int k = sig_max - sig_alpha - sig;
	int wd = sig + k;
	int we = sig_alpha + k;

	q->AddUnaryTerm(Ms, 0, wd);
	q->AddUnaryTerm(Mt, we, 0);

	for (int i = 0; i < points->size(); i++)
	{
		if (state->at(i) == -1)
			continue;

		q->AddPairwiseTerm(Ms, points->at(i).y * imgW + points->at(i).x,
								0,		// E00
								wd,		// E01
								0,		// E10
								0);		// E11


		q->AddPairwiseTerm(Mt, points->at(i).y * imgW + points->at(i).x,
								0,		// E00
								0,		// E01
								we,		// E10
								0);		// E11
	}
}

#define DIFFERENT_STATES 0
#define ALL_0 1
#define ALL_1 2
#define ALL_DONT_CARE 3

// checks the higher order clique whether 0 costs are given in case all pixels are 1 or 0
// accordingly Carsten's non-submodular or Pushmeet submodular constructions are used.
int check_state_homogeneity (vector<int> &state)
{
	int all0 = 1;
	int all1 = 1;
	int all_dont_care = 1;

	for (int i = 0; i < state.size(); i++)
	{
		if (state[i] == 0)
		{
			all1 = 0;
			all_dont_care = 0;
		}
		if (state[i] == 1)
		{
			all0 = 0;
			all_dont_care = 0;
		}
	}

	if (all_dont_care)
		return ALL_DONT_CARE;
	if (all0)
		return ALL_0;
	if (all1)
		return ALL_1;
	return DIFFERENT_STATES;
}

void HOGraphConstruction(QPBO<REAL>* q, vector<CvPoint> *points, vector<int> *state, int PH, int imgW)
{
	if (state->empty())
		return;

	int homogeneity = check_state_homogeneity (*state);

	// if defined we only use Carsten's non-submodular construction
	//#define USE_TYPE1_ONLY
	#ifdef USE_TYPE1_ONLY
		homogeneity = DIFFERENT_STATES;
	#endif

	if (homogeneity == ALL_DONT_CARE)
		return;

	if (homogeneity == ALL_0)
		P3andBeyondConstruction(q, points, state, 0, PH, PH, imgW);
	else if (homogeneity == ALL_1)
		P3andBeyondConstruction(q, points, state, PH, 0, PH, imgW);
	else
		AddType1Construction(q, *points, *state, PH, imgW);

}

// computes the set S_p by intersecting a squared window with the segmentation as described in the paper
void Compute_S_p (vector<CvPoint> *S_p, IplImage *segmentation, CvPoint curpoint, int radius)
{	
	int imgW = segmentation->width;
	int imgH = segmentation->height;

	int x = curpoint.x;
	int y = curpoint.y;

	int centercol[3];
	for (int c = 0; c < 3; c++)
		centercol[c] = (int) (uchar) segmentation->imageData[y * segmentation->widthStep + 3 * x + c];

	for (int wy = y - radius; wy <= y + radius; wy++)
		for (int wx = x - radius; wx <= x + radius; wx++)
		{
			if (wx < 0 || wx >= imgW || wy < 0 || wy >= imgH)
				continue;

			int curcol[3];
			for (int c = 0; c < 3; c++)
				curcol[c] = (int) (uchar) segmentation->imageData[wy * segmentation->widthStep + 3 * wx + c];

			int samecol = 1;
			for (int c = 0; c < 3; c++)
			{
				if (curcol[c] != centercol[c])
					samecol = 0;
			}

			if (samecol)
				S_p->push_back (cvPoint(wx, wy));
		}
}

// computes which 0 or 1 states have to be taken so that the ho-clique is segment consistent
void Compute_Segment_Consistent_State (vector<int> *state, SurfaceModel *valid_model, vector<CvPoint> *S_p, Proposal &proposal1, Proposal &proposal2, int imgW)
{
	for (int i = 0; i < S_p->size(); i++)
	{
		CvPoint point = S_p->at(i);

		SurfaceModel *s1 = proposal1.surfacemodels[point.y * imgW + point.x];
		SurfaceModel *s2 = proposal2.surfacemodels[point.y * imgW + point.x];

		int s1_valid = same_surface_label (s1, valid_model);
		int s2_valid = same_surface_label (s2, valid_model);

		if (s1_valid && s2_valid)
		{
			// -1 is dont care state
			state->push_back (-1);
			//printf ("do not care\n");
		}
		else if (s1_valid && !s2_valid)
			state->push_back (0);
		else if (!s1_valid && s2_valid)
			state->push_back (1);
		else
		{
			state->clear();
			return;
		}
	}
}

// implements the segment consistency term (eq. 7)
void AddHigherOrder (QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, IplImage *segmentation, IplImage *occ_left, int PH, int winsize)
{
	int imgW = segmentation->width;
	int imgH = segmentation->height;

	static vector<CvPoint> *S_p = 0;
	static vector<int> *state_1, *state_2;

	if (!S_p)
	{
		// allocation
		S_p = new vector<CvPoint>;
		S_p->resize (winsize * winsize);

		state_1 = new vector<int>;
		state_1->resize (winsize * winsize);

		state_2 = new vector<int>;
		state_2->resize (winsize * winsize);
	}

	for (int y = 0; y < imgH; y++)
	{
		for (int x = 0; x < imgW; x++)
		{
			S_p->clear();
			state_1->clear();
			state_2->clear();

			SurfaceModel *valid_1 = proposal1.surfacemodels[y * imgW + x];
			SurfaceModel *valid_2 = proposal2.surfacemodels[y * imgW + x];


			Compute_S_p (S_p, segmentation, cvPoint (x,y), winsize / 2);

			// check if both states are the same
			if (same_surface_label (valid_1, valid_2))
			{
				// we need to add the HO only once
				// the current pixel is in donot care mode
				Compute_Segment_Consistent_State (state_1, valid_1, S_p, proposal1, proposal2, imgW);

				int genlikPH = PH;

				HOGraphConstruction(q, S_p, state_1, genlikPH, imgW);
			}
			else
			{
				// for proposal 1
				Compute_Segment_Consistent_State (state_1, valid_1, S_p, proposal1, proposal2, imgW);

				int genlikPH = PH;

				HOGraphConstruction(q, S_p, state_1, genlikPH, imgW);

				// for proposal 2
				Compute_Segment_Consistent_State (state_2, valid_2, S_p, proposal1, proposal2, imgW);

				genlikPH = PH;

				HOGraphConstruction(q, S_p, state_2, genlikPH, imgW);
			}
		}
	}
}



void DrawSegmentBorders (IplImage *img, IplImage *segmentation)
{
	int imgW = img->width;
	int imgH = img->height;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
			for (int c = 0; c < 3; c++)
			{
				int color = (int) (uchar) segmentation->imageData[y * segmentation->widthStep + 3 * x + c];
				color = color / 2 + 122;
				img->imageData[y * img->widthStep + 3 * x + c] = (uchar) color;
			}
}

// evaluates the term and generates an image highlighting segment inconsisten pixels
int Evaluate_SegCons (Proposal &solution, int Pho, int winsize, float scale, vector<IplImage*> segmentations)
{
	if (segmentations.size() == 0)
		return 0;

	int imgW = segmentations[0]->width;
	int imgH = segmentations[0]->height;

	int costs = 0;

	for (int i = 0; i < segmentations.size(); i++)
	{
		IplImage *inconsimg = cvCloneImage (segmentations[i]);
		DrawSegmentBorders (inconsimg, segmentations[i]);
		
		IplImage *segmentation = segmentations[i];

		for (int y = 0; y < imgH; y++)
			for (int x = 0; x < imgW; x++)
			{
				static vector<CvPoint> *S_p = 0;
				if (!S_p)
				{
					S_p = new vector<CvPoint>;
					S_p->resize (winsize * winsize);
				}
				else
					S_p->clear();

				Compute_S_p (S_p, segmentation, cvPoint (x,y), winsize / 2);

				SurfaceModel *centermodel = solution.surfacemodels[y * imgW + x];

				int consistent = 1;

				for (int j = 0; j < S_p->size(); j++)
				{
					CvPoint p = S_p->at(j);
					if (!same_surface_label(centermodel, solution.surfacemodels[p.y * imgW + p.x]))
					{
						consistent = 0;
						break;
					}
				}

				if (!consistent)
				{
					costs += Pho;
					inconsimg->imageData [y * inconsimg->widthStep + 3 * x + 0] = (uchar) 0;
					inconsimg->imageData [y * inconsimg->widthStep + 3 * x + 1] = (uchar) 0;
					inconsimg->imageData [y * inconsimg->widthStep + 3 * x + 2] = (uchar) 0;
				}
			}

			#ifdef SHOWIMAGES
				char winname[1024];
				sprintf (winname, "seginconsistent%d", i + 1);
				cvvNamedWindow (winname, CV_WINDOW_AUTOSIZE);
				cvMoveWindow (winname, (i + 1) * imgW, 2 * imgH);
				cvShowImage (winname, inconsimg);

				char fn[1024];
				sprintf (fn, "..\\results\\%s.png", winname);
				cvSaveImage (fn, inconsimg);

				cvWaitKey(50);
			#endif

			cvReleaseImage (&inconsimg);
	}

	return costs;
}