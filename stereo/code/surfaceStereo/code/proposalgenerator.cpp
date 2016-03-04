#include "proposals.h"

#define USESPLINES 0
#define NOSPLINES 1

// should be 1 or 2
#define MAXSPLINEORDER 1

//#define CONSTANT_PROPOSALS_ONLY
//#define SINGLE_PLANAR_PROPOSAL

ProposalGenerator::ProposalGenerator (char *_testsetname, IplImage *_left, IplImage *_right, int _maxdisp, float _scale, int _iterations)
{
	testsetname = _testsetname;
	left = _left;
	right = _right;
	maxdisp = _maxdisp;
	scale = _scale;
	iterations = _iterations;

	id = 0;
	curit = 0;

	// init the parameters of the different proposal types described in section 4.2
	InitMeanshiftProposals();
	InitConstantProposals();
	InitRefitProposals();
	kmeanscounter = 0;
	planemergecounter = 0;
	segmentdilationcounter = 0;
}

int ProposalGenerator::GetNewId ()
{
	return id++;
}

////////////////////////////////////////////////////////
//		Proposal Type 1 Seg + model Fitting
void ProposalGenerator::InitMeanshiftProposals ()
{
	vector<SegParam> segparams;
	SegParam segparam;

	#ifndef SINGLE_PLANAR_PROPOSAL
	segparam.SpatialBandwidth = 8.0; segparam.RangeBandwidth = 7.5; segparam.MinimumRegionsArea = 80; segparams.push_back (segparam);
	segparam.SpatialBandwidth = 10.0; segparam.RangeBandwidth = 15.0; segparam.MinimumRegionsArea = 300; segparams.push_back (segparam);
	segparam.SpatialBandwidth = 15.0; segparam.RangeBandwidth = 10.0; segparam.MinimumRegionsArea = 300; segparams.push_back (segparam);
	segparam.SpatialBandwidth = 20.0; segparam.RangeBandwidth = 25.0; segparam.MinimumRegionsArea = 600; segparams.push_back (segparam);
	segparam.SpatialBandwidth = 25.0; segparam.RangeBandwidth = 20.0; segparam.MinimumRegionsArea = 600; segparams.push_back (segparam);
	#endif

	vector<STParam> stparams;
	STParam stparam;

	stparam.P1 = 40;  stparam.P2 = 80;  stparams.push_back (stparam);
	#ifndef SINGLE_PLANAR_PROPOSAL
	stparam.P1 = 30;  stparam.P2 = 80;  stparams.push_back (stparam);
	stparam.P1 = 50;  stparam.P2 = 100;  stparams.push_back (stparam);
	#endif

	vector<float> plane_thresholds;
	plane_thresholds.push_back (1.f);

	int splinenumber = 0;
	#ifndef SINGLE_PLANAR_PROPOSAL
		for (splinenumber = 0; splinenumber <= MAXSPLINEORDER; splinenumber++)
	#endif
		for (int i = 0; i < segparams.size(); i++)
			for (int j = 0; j < stparams.size(); j++)
				for (int k = 0; k < plane_thresholds.size(); k++)
				{
					PlaneProposalParm ppp;
					
					ppp.proposalid = GetNewId();
					
					ppp.segparam = segparams[i];
					ppp.stparam = stparams[j];
					ppp.plane_threshold = plane_thresholds[k];
					ppp.splinenumber = splinenumber;

					ppp.processed = 0;

					planeparams.push_back (ppp);
				}
}

Proposal *ProposalGenerator::GenerateMeanshiftProposal(int use_splines)
{
	Proposal *proposal = 0;	

	// plane params contain all parameter settings for the Segmentation + Model Fitting proposals (see paper)
	for (int i = 0; i < planeparams.size(); i++)
	{
		if (proposal)
			break;

		PlaneProposalParm ppp = planeparams[i];
		if (ppp.processed || (ppp.splinenumber > 0 && use_splines == NOSPLINES))
			continue;

		printf ("Meanshift Proposal / Spline Order %d\n", ppp.splinenumber);

		proposal = new Proposal (left, right, maxdisp, scale, ppp.proposalid);

		// generate proposal given the parameters of meanshift segmentation and simple tree algorithm
		proposal->gernerate_meanshift_proposal (testsetname, 
			ppp.segparam.SpatialBandwidth, ppp.segparam.RangeBandwidth, ppp.segparam.MinimumRegionsArea,
			ppp.stparam.P1, ppp.stparam.P2, ppp.splinenumber,
			ppp.plane_threshold);

		planeparams[i].processed = 1;

	}

	return proposal;
}

////////////////////////////////////////////////////////
//		Proposal Type 2 Fronto-parallel planes
void ProposalGenerator::InitConstantProposals ()
{
	for (int d = 0; d <= maxdisp; d++)
	{
		ConstantProposalParm param;

		param.proposalid = GetNewId();

		param.d = (float) d;
		param.processed = 0;

		constantparams.push_back (param);
	}
}

Proposal *ProposalGenerator::GenerateConstantProposal()
{
	Proposal *proposal = 0;	

	for (int i = 0; i < constantparams.size(); i++)
	{
		if (proposal)
			break;

		ConstantProposalParm param = constantparams[i];
		if (param.processed)
			continue;

		printf ("Constant Proposal disparity %.1f\n", param.d);

		proposal = new Proposal (left, right, maxdisp, scale, param.proposalid);

		proposal->gernerate_constant_proposal(0.f, 0.f, param.d);

		constantparams[i].processed = 1;
	}

	return proposal;
}

////////////////////////////////////////////////////////
//	Proposal Refit the current surfaces

#define USECURRENTDISP -1

void ProposalGenerator::InitRefitProposals ()
{
	vector<STParam> stparams;
	STParam stparam;

	stparam.P1 = USECURRENTDISP;  stparam.P2 = USECURRENTDISP;  stparams.push_back (stparam);
	stparam.P1 = 40;  stparam.P2 = 80;  stparams.push_back (stparam);
	stparam.P1 = 30;  stparam.P2 = 80;  stparams.push_back (stparam);
	stparam.P1 = 50;  stparam.P2 = 100;  stparams.push_back (stparam);	
	stparam.P1 = 60;  stparam.P2 = 120;  stparams.push_back (stparam);	
	stparam.P1 = 30;  stparam.P2 = 150;  stparams.push_back (stparam);

	for (int i = 0; i < stparams.size(); i++)
		for (int numsplines = 0; numsplines <= MAXSPLINEORDER; numsplines++)
		{
			RefitParam param;
			param.stparam = stparams[i];
			param.numsplines = numsplines;
			param.processed = 0;
			refitparams.push_back (param);
		}
}

Proposal *ProposalGenerator::GenerateRefitProposal(Proposal *cursolution)
{
	Proposal *proposal = 0;	

	for (int i = 0; i < refitparams.size(); i++)
	{
		if (proposal)
			break;

		RefitParam p = refitparams[i];

		if (p.processed)
			continue;

		printf ("Refit Proposal / Spline Order %d\n", p.numsplines);

		int proposalid = GetNewId();
		proposal = new Proposal (left, right, maxdisp, scale, proposalid);


		IplImage *disp;
		if (p.stparam.P1 == USECURRENTDISP)
			disp = cursolution->plot_proposal(scale);
		else
			disp = ExternDisparityMap (testsetname, left, right, maxdisp, scale, p.stparam.P1, p.stparam.P2);


		IplImage *segmentation = cursolution->plot_surface_ids();

		proposal->gernerate_proposal_via_disp_and_segmentation (disp, segmentation, p.numsplines, 1.0);

		cvReleaseImage (&disp);
		cvReleaseImage (&segmentation);

		refitparams[i].processed = 1;

	}

	return proposal;
}


////////////////////////////////////////////////////////
//	Proposal Type 3 Expansion of largest planes

void ProposalGenerator::InitSingleSurfaceExpansion (Proposal *proposal)
{
	map<SurfaceModel*, int, lessthanSurfaceModel> surfaceareas;
	map<SurfaceModel*, int, lessthanSurfaceModel>::iterator mapit;

	int imgW = proposal->imgW;
	int imgH = proposal->imgH;

	// compute area of each surface
	for (int i = 0; i < imgW * imgH; i++)
	{
		SurfaceModel *cur = proposal->surfacemodels[i];

		mapit = surfaceareas.find (cur);

		if (mapit == surfaceareas.end())
			surfaceareas[cur] = 1;
		else
			surfaceareas[cur] = surfaceareas[cur] + 1;
	}

	// find surfaces of large area
	for (mapit = surfaceareas.begin(); mapit != surfaceareas.end(); mapit++)
	{
		int area = mapit->second;

		if (area > imgW * imgH / 200)
		{
			SingleSurfaceParam ssp;

			ssp.surface = mapit->first->Clone();
			ssp.processed = 0;

			singlesurfaces.push_back(ssp);
		}
	}
}

Proposal *ProposalGenerator::GenerateSingleSurfaceProposal()
{
	Proposal *proposal = 0;	

	for (int i = 0; i < singlesurfaces.size(); i++)
	{
		if (proposal)
			break;

		SingleSurfaceParam ssp = singlesurfaces[i];
		if (ssp.processed)
			continue;

		printf ("Single Surface Expansion\n");

		proposal = new Proposal (left, right, maxdisp, scale, ssp.surface->proposalid);

		int imgW = proposal->imgW;
		int imgH = proposal->imgH;

		for (int i = 0; i < imgW * imgH; i++)
			proposal->surfacemodels[i] = ssp.surface;

		singlesurfaces[i].processed = 1;
	}

	return proposal;
}

////////////////////////////////////////////////////////
//	Proposal Type 4 Segmentation via kmeans
Proposal *ProposalGenerator::GenerateKMeansProposal(Proposal *cursolution)
{
	if (kmeanscounter >= 10)
		return 0;

	kmeanscounter++;

	int splinenumber = rand() % (MAXSPLINEORDER + 1);
	int NumberOfClusters = rand() % 40 + 5;
	printf ("KMeans Proposal / Spline Order %d NumberOfClusters %d\n", splinenumber, NumberOfClusters);


	//IplImage *disp = cvLoadImage ("data\\fused.png", 0);
	IplImage *disp = cursolution->plot_proposal(scale);


	printf ("NumberOfClusters %d\t", NumberOfClusters);

	IplImage *segmentation = kmeans_segmentation (disp, NumberOfClusters);

	int proposalid = GetNewId();
	printf ("proposalid %d\n", proposalid);

	Proposal *proposal = new Proposal (left, right, maxdisp, scale, proposalid);
	proposal->gernerate_proposal_via_disp_and_segmentation (disp, segmentation, splinenumber, 1.0);

	cvReleaseImage (&disp);
	cvReleaseImage (&segmentation);

	return proposal;
}

Proposal *ProposalGenerator::GenerateSegmentDilationProposal(Proposal *cursolution)
{
	int toogle = rand();
	if (!cursolution || toogle % 5 != 0)
		return 0;

	printf ("Segment Dilation Proposal\n");

	Proposal *prop = segmentdilation (cursolution);

	return prop;
}


void ProposalGenerator::Reset()
{
	#ifdef CONSTANT_PROPOSALS_ONLY
		for (int i = 0; i < constantparams.size(); i++)
			constantparams[i].processed = 0;
	#endif


	for (int i = 0; i < refitparams.size(); i++)
		refitparams[i].processed = 0;

	singlesurfaces.clear();
	kmeanscounter = 0;
	planemergecounter = 0;
	segmentdilationcounter = 0;
}




Proposal *ProposalGenerator::GetProposal(Proposal *cursolution)
// implements the proposal sequence described in section 4.2.
{
	Proposal *proposal = 0;

	// a call to a proposal type returns 0 if all proposals of this type have already been consumed

	while (curit < iterations)
	{
		printf ("\n\n********************\n");
		printf ("iteration %d of %d\n", curit, iterations);
		printf ("********************\n\n\n");

		// Surface Dilation Proposals
		// these are called randomly
		proposal = GenerateSegmentDilationProposal(cursolution);
		if (proposal)
			return proposal;

		// Segmentation + Model Fitting Proposals of the paper
		proposal = GenerateMeanshiftProposal(USESPLINES);
		if (proposal)
			return proposal;

		// Single Surface Proposals
		if (singlesurfaces.empty())
			InitSingleSurfaceExpansion(cursolution);
		proposal = GenerateSingleSurfaceProposal();
		if (proposal)
			return proposal;

		// Fronto-Parallel Proposals
		// these consume a lot of time
		// proposal = GenerateConstantProposal();
		// if (proposal)
		//	return proposal;

		// Refit Proposals
		proposal = GenerateRefitProposal(cursolution);
		if (proposal)
			return proposal;

		// KMeans Proposals
		proposal = GenerateKMeansProposal(cursolution);
		if (proposal)
			return proposal;

		// if all proposals have been consumed 
		// rest and increment iteration count
		if (!proposal)
		{
			Reset();
			curit++;
		}
	}

	return proposal;
}

Proposal *ProposalGenerator::GetPlaneProposal()
{
	Proposal *proposal = 0;

	while (curit < iterations)
	{
		proposal = GenerateMeanshiftProposal(NOSPLINES);
		if (proposal)
			return proposal;

		if (!proposal)
		{
			Reset();
			curit++;
		}
	}

	return proposal;
}