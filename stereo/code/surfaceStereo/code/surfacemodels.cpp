#include "surfacemodels.h"

float Plane::PointDisp (CvPoint point, float scale)
{
	float d = A * point.x + B * point.y + C;

	//int d_int = (int) (d * scale + 0.5);

	//d = (float) d_int / scale;

	return d;
}

SurfaceModel *Plane::Clone()
{
	Plane *plane = new Plane();

	plane->A = A;
	plane->B = B;
	plane->C = C;
	plane->proposalid = proposalid;
	plane->segid = segid;

	return (SurfaceModel*) plane;
}

vector<float> Plane::SupportVals (CvPoint point, int winSize, float scale)
{
	vector<float> disps;

	int radius = winSize / 2;

	for (int y = point.y - radius; y <= point.y + radius; y++)
		for (int x = point.x - radius; x <= point.x + radius; x++)
			disps.push_back (PointDisp (cvPoint (x,y), scale));

	return disps;
}

int Plane::which_type ()
{
	if (A == 0.f && B == 0.f)
		return IS_CONSTANT_SURFACE;

	return IS_PLANE;
}

void Plane::Save (FILE *fp)
{
	fprintf (fp, "%d %d %d %f %f %f\n", which_type(), proposalid, segid, A, B, C);
}

int Plane::is_valid()
{
	if (A > 0.5f)
		return 0;
	else
		return 1;
}