#ifndef FUSION_H
#define FUSION_H

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "QPBO.h"
#include "proposals.h"

#include <stdio.h>

#include <vector>
using namespace std;

typedef int REAL;

#define CURVRAD 5

// determines wheter images are shown or not
#define SHOWIMAGES

/////////////////////////////////////////////////////////
//		fusion.cpp
/////////////////////////////////////////////////////////
Proposal fuse(ProposalGenerator &generator, 
			  int OccPen, int SmoothPen, int HOPen, vector<int> segimages, 
			  int SparsePen, float SurfComplexityPen, int winsize,
			  int *dsi, int maxdisp, float scale, 
			  char *name, IplImage *left, IplImage *occ_left, char *disp_fn);

/////////////////////////////////////////////////////////
//		fusion_matchcosts.cpp
/////////////////////////////////////////////////////////
int MatchingCosts (CvPoint p, Proposal &proposal, int *dsi, int OccPen);

/////////////////////////////////////////////////////////
//		fusion_unaries.cpp
/////////////////////////////////////////////////////////
void AddUnaries (kolmogorov::qpbo::QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, int *dsi);
int EvaluateUnaryCosts (Proposal &solution, int *dsi);

void AddSurfaceComplexityUnary (kolmogorov::qpbo::QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, int SurfComplexityPen);
int EvaluateSurfaceComplexityUnary (Proposal &solution, int SurfComplexityPen);

void AddCurvatureUnary (kolmogorov::qpbo::QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, int CurvPen);
int EvaluateCurvatureUnary (Proposal &solution, int CurvPen);

/////////////////////////////////////////////////////////
//		fusion_binaries.cpp
/////////////////////////////////////////////////////////
void AddBinaries_SurfaceLabels (kolmogorov::qpbo::QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, int SmoothPen);
int EvaluateBinaries_SurfaceLabels (Proposal &solution, int SmoothPen);
int same_surface_label (SurfaceModel *s1, SurfaceModel *s2);

/////////////////////////////////////////////////////////
//		fusion_occ.cpp
/////////////////////////////////////////////////////////
void AddDataTermWithOcc (kolmogorov::qpbo::QPBO<REAL> *q, Proposal *proposal1, Proposal *proposal2, int *dsi, int OccPen, vector<int> *oldlabels);
int Evaluate_DataWithOcc (kolmogorov::qpbo::QPBO<REAL> *q, Proposal *proposal1, Proposal *proposal2);
void ShowOcclusionGraph (kolmogorov::qpbo::QPBO<REAL> *q, Proposal *proposal1, Proposal *proposal2, float scale, char *fn = 0);
void Getoldlabels (kolmogorov::qpbo::QPBO<REAL> *q, vector<int> *&oldlabels);

/////////////////////////////////////////////////////////
//		fusion_ho.cpp
/////////////////////////////////////////////////////////
void AddHigherOrder (kolmogorov::qpbo::QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, IplImage *segmentation, IplImage *occ_left, int PH, int winsize);
int Evaluate_SegCons (Proposal &solution, int Pho, int winsize, float scale, vector<IplImage*> segmentations);

/////////////////////////////////////////////////////////
//		fusion_ncc.cpp
/////////////////////////////////////////////////////////
float ncc (Proposal &proposal, IplImage *occ_left, vector<CvPoint> &S_p, SurfaceModel *sm);
float cenusus (Proposal &proposal, IplImage *occ_left, CvPoint center, vector<CvPoint> &S_p, SurfaceModel *sm);
float AVG_SAD (Proposal &proposal, IplImage *occ_left, vector<CvPoint> &S_p, SurfaceModel *sm);

/////////////////////////////////////////////////////////
//		fusion_sparsity.cpp
/////////////////////////////////////////////////////////
void AddSparsityPrior(kolmogorov::qpbo::QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, int Psparse, float SurfComplexityPen);
int Evaluate_Sparsity (Proposal &solution, int Psparse, float SurfComplexityPen);

/////////////////////////////////////////////////////////
//		fusion_utils.cpp
/////////////////////////////////////////////////////////
void ShowImages (kolmogorov::qpbo::QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, Proposal &fused, int OccPen, int HOPen, char *disp_fn);
void showUnlabeled (kolmogorov::qpbo::QPBO<REAL>* q, IplImage *left);
int numUnlabeled (kolmogorov::qpbo::QPBO<REAL>* q, int imgW, int imgH);
float ComputeCurvature (CvPoint p, SurfaceModel *sm);
float ComputeCurvatureSurfaceBorders (CvPoint p1, CvPoint p2, SurfaceModel *sm1, SurfaceModel *sm2, int imgW, int imgH);

void RoundDisp (IplImage *fusedimg);

#endif