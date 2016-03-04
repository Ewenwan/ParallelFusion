#include "fusion.h"

// function looks up the matching costs from the cost volume
int MatchingCosts (CvPoint p, Proposal &proposal, int *dsi, int OccPen)
{
	int imgW = proposal.imgW;
	float scale = proposal.scale;
	int maxdisp = proposal.maxdisp;

	float float_d = proposal.surfacemodels[p.y * imgW + p.x]->PointDisp(cvPoint (p.x,p.y), scale);

	if (float_d < 0 || float_d > (float) maxdisp)
		return 999;

	int int_d = (int) (float_d + 0.5f);

	int *ptr = dsi + p.y * imgW * (maxdisp + 1) + p.x * (maxdisp + 1);
	int c = ptr[int_d];

	//return c;
	if (OccPen > 0)
		return min(c, OccPen - 1);
	else
		return c;
}