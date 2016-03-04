#ifndef PROPOSAL_H
#define PROPOSAL_H

#include "externmatcher.h"
#include "segment.h"
#include "modelfitting.h"
#include "kmeans.h"

#include <vector>
#include <algorithm>
#include <set>
#include <map>
using namespace std;

typedef class Proposal
{
public:
	vector<SurfaceModel*> surfacemodels;
	int imgW;
	int imgH;

	int maxdisp;
	float scale;
	IplImage *left, *right;


	Proposal (IplImage *_left, IplImage *_right, int _maxdisp, float _scale, int _proposalid);

	Proposal (Proposal *p);

	~Proposal();

	int get_surface_cnt();

	int get_spline_cnt();

	void gernerate_meanshift_proposal (char *imgname, float MS_SpatialBandwidth, float MS_RangeBandwidth, float MS_MinimumRegionsArea, int ST_P1, int ST_P2, int splinenumber, float plane_threshold);

	void gernerate_proposal_via_disp_and_segmentation (IplImage *disp, IplImage *segmentation, int splinenumber, float plane_threshold);

	void gernerate_planemerge_proposal (Proposal *src_proposal, float planesim_threshold);

	void gernerate_constant_proposal (float A, float B, float C);

	IplImage *plot_proposal (float scale);

	IplImage *plot_height_lines (IplImage *disp);

	IplImage *plot_surface_ids ();

	IplImage *plot_spline_order ();

	void Merge (Proposal *proposal1, Proposal *proposal2, vector<int> binlabels);

	void Save (char *fn);

	void Load (char *fn);

	int proposalid;

private:

};

typedef struct {
	float SpatialBandwidth; 
	float RangeBandwidth; 
	float MinimumRegionsArea;
} SegParam;

typedef struct {
	int P1; 
	int P2; 
} STParam;

typedef struct {
	SegParam segparam;
	STParam stparam;
	float plane_threshold;
	int splinenumber;
	int processed;
	int proposalid;
	int segid;
} PlaneProposalParm;

typedef struct {
	float d;
	int processed;
	int proposalid;
} ConstantProposalParm;

typedef struct {
	SurfaceModel *surface;
	int processed;
} SingleSurfaceParam;

typedef struct {
	STParam stparam;
	int numsplines;
	int processed;
} RefitParam;

typedef class ProposalGenerator
{
public:
	ProposalGenerator (char *_testsetname, IplImage *_left, IplImage *_right, int _maxdisp, float _scale, int _iterations);

	Proposal *GetProposal(Proposal *cursolution);

	Proposal *GetPlaneProposal();

private:
	int id;
	int curit;

	char *testsetname;
	IplImage *left, *right;
	int maxdisp;
	float scale;
	int iterations;

	vector<PlaneProposalParm> planeparams;
	vector<ConstantProposalParm> constantparams;
	vector<SingleSurfaceParam> singlesurfaces;
	vector<RefitParam> refitparams;
	int kmeanscounter;
	int planemergecounter;
	int segmentdilationcounter;

	int GetNewId ();

	void InitMeanshiftProposals ();

	void InitConstantProposals ();

	void InitRefitProposals ();

	void InitSingleSurfaceExpansion (Proposal *proposal);

	Proposal *GenerateMeanshiftProposal(int usespline);

	Proposal *GenerateConstantProposal();

	Proposal *GenerateRefitProposal(Proposal *cursolution);

	Proposal *GenerateSingleSurfaceProposal();

	Proposal *GenerateKMeansProposal(Proposal *cursolution);

	Proposal *GenerateSegmentDilationProposal(Proposal *cursolution);

	Proposal *GeneratePlaneMergingProposal(Proposal *cursolution);

	void Reset();
	
};

IplImage *Median_Disparity_Map (vector<IplImage*> disp_maps);

Proposal *segmentdilation (Proposal *cursolution);

#endif