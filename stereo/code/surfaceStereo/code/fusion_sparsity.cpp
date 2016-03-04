#include <map>


#include "fusion.h"

using namespace std;
using namespace kolmogorov::qpbo;

#define MYINFINTIY 100000

struct MapEntry
{
	CvPoint point;
	int state; // state at which the surface is valid (0 or 1)
};

set<SurfaceModel*, lessthanSurfaceModel> proposal1surfaces;
map<SurfaceModel*, vector<MapEntry>, lessthanSurfaceModel> surfmap;


int GetSparsenessCosts (SurfaceModel *sm, int Psparse, float SurfComplexityPen, float scale)
{
	const int LARGEVAL = 500000;

	if (!sm->is_valid())
		return LARGEVAL;

	if (sm->splineorder() == 0)
		return Psparse;

	if (sm->splineorder() == 1)
	{
		// could be adjusted to make splines more expensive than planes
		return (int) ((float) Psparse * 1.0f);
	}

	printf ("spline of order > 1 detected (sparsity term)");
	return LARGEVAL;
}

void Remove01Entries (Proposal &proposal1, Proposal &proposal2)
{
	int imgW = proposal1.imgW;
	int imgH = proposal1.imgH;

	map<SurfaceModel*, vector<MapEntry>, lessthanSurfaceModel>::iterator mapit;

	for (int i = 0; i < imgW * imgH; i++)
	{
		SurfaceModel *sm1 = proposal1.surfacemodels[i];
		SurfaceModel *sm2 = proposal2.surfacemodels[i];

		if (sm1->proposalid == sm2->proposalid &&
			sm1->segid == sm2->segid)
		{
			mapit = surfmap.find (sm1);
			if (mapit != surfmap.end())
			{
				surfmap.erase (mapit);
			}
		}
	}
}


// adds MDL term of (eq. (9)). Uses the construction of Hoeim
void AddSparsityPrior(QPBO<REAL>* q, Proposal &proposal1, Proposal &proposal2, int Psparse, float SurfComplexityPen)
{
	int imgW = proposal1.imgW;
	int imgH = proposal1.imgH;

	proposal1surfaces.clear();
	surfmap.clear();

	for (int i = 0; i < 2; i++)
	{
		Proposal *prop;

		if (i == 0)
			prop = &proposal1;
		else 
			prop = &proposal2;

		for (int y = 0; y < imgH; y++)
			for (int x = 0; x < imgW; x++)
			{
				MapEntry mapentry;
				mapentry.point = cvPoint (x,y);
				mapentry.state = i;

				SurfaceModel *cursurf = prop->surfacemodels[y * imgW + x];

				map<SurfaceModel*, vector<MapEntry>, lessthanSurfaceModel>::iterator mapit;
				mapit = surfmap.find (cursurf);

				if (mapit != surfmap.end())
				{
					mapit->second.push_back (mapentry);
				}
				else
				{
					vector<MapEntry> vec;
					vec.push_back (mapentry);
					surfmap [cursurf] = vec;

					if (i == 0)
						proposal1surfaces.insert(cursurf);
				}
			}
	}

	// just for speed up
	Remove01Entries (proposal1, proposal2);

	map<SurfaceModel*, vector<MapEntry>, lessthanSurfaceModel>::iterator mapit;
	set<SurfaceModel*, lessthanSurfaceModel>::iterator setit;

	for (mapit = surfmap.begin(); mapit != surfmap.end(); mapit++)
	{
		vector<MapEntry> entries = mapit->second;

		int instance_node = q->AddNode(1);

		int pen = GetSparsenessCosts (mapit->first, Psparse, SurfComplexityPen, proposal1.scale);

		// for surfaces of the old proposal (proposal1) I switch the meaning of 0 and 1
		// to ensure that labelling <0,0,...0> does not include infinite edges (QPBOI problem)
		int switchmeaning;
		// check if surface is part of proposal1
		SurfaceModel *sm = mapit->first;
		setit = proposal1surfaces.find(sm);
		if (setit == proposal1surfaces.end())
			switchmeaning = 0;
		else 
			switchmeaning = 1;

		if (!switchmeaning)
			q->AddUnaryTerm(instance_node, 0, pen);
		else
			q->AddUnaryTerm(instance_node, pen, 0);

		for (int i = 0; i < entries.size(); i++)
		{
			CvPoint p = entries[i].point;
			int idx = p.y * imgW + p.x;
			int state = entries[i].state;

			if (!switchmeaning)
			{
				if (state == 1)
				{
					q->AddPairwiseTerm(idx, instance_node,
										0,			// E00
										0,			// E01
										MYINFINTIY,	// E10
										0);			// E11
				}
				if (state == 0)
				{
					q->AddPairwiseTerm(idx, instance_node,
										MYINFINTIY,	// E00
										0,			// E01
										0,			// E10
										0);			// E11
				}
			}
			else
			{
				if (state == 1)
				{
					q->AddPairwiseTerm(idx, instance_node,
										0,			// E00
										0,			// E01
										0,			// E10
										MYINFINTIY);	// E11
				}
				if (state == 0)
				{
					q->AddPairwiseTerm(idx, instance_node,
										0,			// E00
										MYINFINTIY,	// E01
										0,			// E10
										0);			// E11
				}
			}

		}
	}
}


// evaluates MDL term of (eq. (9))
int Evaluate_Sparsity (Proposal &solution, int Psparse, float SurfComplexityPen)
{
	int imgW = solution.imgW;
	int imgH = solution.imgH;

	map<SurfaceModel*, vector<MapEntry>, lessthanSurfaceModel> surfmap;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			SurfaceModel *cursurf = solution.surfacemodels[y * imgW + x];

			map<SurfaceModel*, vector<MapEntry>, lessthanSurfaceModel>::iterator mapit;
			mapit = surfmap.find (cursurf);

			if (mapit == surfmap.end())
			{
				vector<MapEntry> vec;
				surfmap [cursurf] = vec;
			}
		}

	int costs = 0;

	map<SurfaceModel*, vector<MapEntry>, lessthanSurfaceModel>::iterator mapit;
	for (mapit = surfmap.begin(); mapit != surfmap.end(); mapit++)
	{
		costs += GetSparsenessCosts (mapit->first, Psparse, SurfComplexityPen, solution.scale);
	}

	return costs;
}