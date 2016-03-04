#include "fusion.h"

void LoadSegmentationImages (char *name, vector<IplImage*> *segmentations, vector<int> segindices)
// load precompute meanshift segmentations
// if you want to use different images use edison and call the segmented color image ms3_segm.png
// store the file in directory name
{
	char segname[1024];

	for (int i = 0; i < segindices.size(); i++)
	{
		int idx = segindices[i];

		if (idx == 1)
		{
			sprintf (segname, "..\\data\\segmentation\\%s\\ms1_segm.png", name);
			segmentations->push_back(cvLoadImage (segname));
		}
		if (idx == 2)
		{
			sprintf (segname, "..\\data\\segmentation\\%s\\ms2_segm.png", name);
			segmentations->push_back(cvLoadImage (segname));
		}
		if (idx == 3)
		{
			sprintf (segname, "..\\data\\segmentation\\%s\\ms3_segm.png", name);
			segmentations->push_back(cvLoadImage (segname));
		}
		if (idx == 4)
		{
			sprintf (segname, "..\\data\\segmentation\\%s\\ms4_segm.png", name);
			segmentations->push_back(cvLoadImage (segname));
		}
	}
}


Proposal *GetSolutionFromGraph (QPBO<REAL>* q, Proposal *proposal1, Proposal *proposal2)
{
	int imgW = proposal1->imgW;
	int imgH = proposal1->imgH;

	vector<int> binlabels;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			int x_i = q->GetLabel(y * imgW + x);

			if (x_i == 0) 
				binlabels.push_back(0);

			if (x_i == 1)
				binlabels.push_back(1);

			if (x_i < 0)
				binlabels.push_back(0);
		}

	Proposal *fused = new Proposal (proposal1);
	fused->Merge (proposal1, proposal2, binlabels);

	return fused;
}



// computes the optimal fusion of propsal1 and proposal2 such that energy (1) is minimized
Proposal *fuse2solutions (Proposal *proposal1, Proposal *proposal2, 
						  int OccPen, int SmoothPen, int HOPen, vector<int> segindices,
						  int SparsePen, float SurfComplexityPen, int winsize,
						  char *name, IplImage *left, IplImage *occ_left, int *dsi, char *disp_fn)
{
	int imgW = proposal1->imgW;
	int imgH = proposal1->imgH;

	int maxdisp = proposal1->maxdisp;
	float scale = proposal1->scale;

	// proposal counter
	static int it = 0;
	it++;

	printf ("//////////////////////////////////\n");
	printf ("//		proposal %d\t//\n", it);
	printf ("//////////////////////////////////\n");

	// energy in the previous step
	static int prev_e = 99999999999;

	// record labels
	// workaround which allows using QPBOI
	// QPBOI assumes that by setting all nodes in the graph to label 0 leads to obtaining proposal 1
	// this is not true because occlusion nodes may also be set to 1 in proposal 1
	static vector<int> *oldlabels = 0;
	if (!oldlabels)
	{
		oldlabels = new vector<int>;
		for (int i = 0; i < 3 * imgW * imgH; i++)
			oldlabels->push_back(0);
	}

	// number of pixel nodes
	int nNodes = imgW * imgH;

	// allocate the graph
	static QPBO<REAL>* q = 0;
	if (!q)
		q = new QPBO<REAL>(7 * nNodes, 100 * nNodes); // max number of nodes & edges

	// construct the graph that represent energy (1) under fusion moves

	// add the data term of eq (2)
	if (OccPen <= 0)
	{
		// if OccPen is set to 0 we use a standard unary without occlusion handling
		printf ("adding data term (no occlusions)\n");
		AddUnaries (q, *proposal1, *proposal2, dsi);
	}
	else
	{
		// eq (2)
		printf ("adding data term (with occlusions)\n");
		AddDataTermWithOcc (q, proposal1, proposal2, dsi, OccPen, oldlabels);
	}

	// penalty on curvature in eq. (8)
	if (SurfComplexityPen > 0.f)
	{
		printf ("adding penalty on curvature\n");
		AddCurvatureUnary (q, *proposal1, *proposal2, (int) SurfComplexityPen);
	}

	// pairwise smoothness term of eq. (5)
	if (SmoothPen > 0)
	{
		printf ("adding pairwise smoothness term\n");
		AddBinaries_SurfaceLabels (q, *proposal1, *proposal2, SmoothPen);
	}

	// soft segmentation term of eq. (7)
	static vector<IplImage*> *segmentations = 0;
	if (HOPen > 0)
	{
		printf ("adding segment consistency term (higher order)\n");

		if (!segmentations)
		{
			// if not already done load the precomputed mean shift segmentations used to obtain $S_p$
			// this can also be multiple images
			segmentations = new vector<IplImage*>;
			LoadSegmentationImages (name, segmentations, segindices);
		}

		// add the segmentation consistency term for each provided color segmentation
		for (int i = 0; i < segmentations->size(); i++)
		{
			AddHigherOrder (q, *proposal1, *proposal2, (*segmentations)[i], occ_left, HOPen, winsize);
			printf ("\n");
		}
	}

	// mdl term of eq (9)
	if (SparsePen > 0)
	{
		printf ("adding sparsity term\n");
		AddSparsityPrior(q, *proposal1, *proposal2, SparsePen, SurfComplexityPen);
	}

	/////////////////////////////////////////////////////
	//					QBPO                           //
	printf ("Solving...\n");

	// run QPBO
	q->MergeParallelEdges();
	q->Solve();
	q->ComputeWeakPersistencies();

	// the number of pixels left unlabelled by QPBO
	int numunlab = numUnlabeled(q, imgW, imgH);
	showUnlabeled (q, left);

	/////////////////////////////////////////////////////
	// F step - most parts of the graph (except aux    //
	// nodes) are defined so that 0 gives old solution //
	for (int i = 0; i < q->GetNodeNum(); i++)
	{
		int label = q->GetLabel (i);
		if (label >= 0)
			q->SetLabel(i, label);
		else
			q->SetLabel(i, 0);	
	}



	int e_check = q->ComputeTwiceEnergy() / 2;
	printf (" e old labels:\t %d\n", e_check);

	// run the improvement step (QPBOI)
	int noimp_it = 0;
	while (1 && numunlab > 0)
	{
		bool improvement = q->Improve();
		if (improvement == true)
		{
			printf ("+");
			noimp_it = 0;
		}
		else
		{
			printf ("-");
			noimp_it++;
		}

		int e_check = q->ComputeTwiceEnergy() / 2;
		printf (" e after I:\t %d\n", e_check);

		if (noimp_it > 0)
			break;
	}

	// get the new proposal given the binary labelling
	Proposal *fused = GetSolutionFromGraph (q, proposal1, proposal2);

	/////////////////////////////////
	//		output
	ShowImages (q, *proposal1, *proposal2, *fused, OccPen, HOPen, disp_fn);

	// compute the new energy (used for double-checking, could be skipped)
	printf ("============== energy ==============\n");
	int e = 0;

	if (OccPen <= 0)
	{
		int e_data = EvaluateUnaryCosts (*fused, dsi);
		printf ("e_data =\t %d\n", e_data);
		e += e_data;
	}
	else
	{
		int e_data = Evaluate_DataWithOcc (q, proposal1, proposal2);
		printf ("e_dataocc =\t %d\n", e_data);
		e += e_data;
	}

	if (SurfComplexityPen > 0.f)
	{
		int e_curv = EvaluateCurvatureUnary (*fused, (int) SurfComplexityPen);
		printf ("e_curv =\t %d\n", e_curv);
		e += e_curv;
	}

	if (SmoothPen > 0)
	{
		int e_smooth = EvaluateBinaries_SurfaceLabels (*fused, SmoothPen);
		printf ("e_smooth =\t %d\n", e_smooth);
		e += e_smooth;
	}

	if (HOPen > 0)
	{
		int e_segcons = Evaluate_SegCons (*fused, HOPen, winsize, scale, *segmentations);
		printf ("e_segcons =\t %d\n", e_segcons);
		e += e_segcons;
	}

	if (SparsePen > 0)
	{
		int e_sparsity = Evaluate_Sparsity (*fused, SparsePen, SurfComplexityPen);
		printf ("e_sparsity =\t %d\t (surface count %d / splines %d)\n", e_sparsity, fused->get_surface_cnt(), fused->get_spline_cnt());
		e += e_sparsity;
	}

	printf ("e = %d\n", e);
	printf ("====================================\n");

	// check if energy is raising (this would mean we have an error in the graph construction)
	if (prev_e < e)
	{
		printf ("====================================\n");
		printf ("====================================\n");
		printf ("energy rising %d %d\n", prev_e, e);
		printf ("====================================\n");
		printf ("====================================\n");
		//cvWaitKey(0);
	}

	// new energy becomes old one
	prev_e = e;

	// workaround for the QPBOI problem described above
	if (OccPen > 0)
		Getoldlabels (q, oldlabels);

	q->Reset();

	return fused;
}




Proposal fuse(ProposalGenerator &generator, 
			  int OccPen, int SmoothPen, int HOPen, vector<int> segimages, 
			  int SparsePen, float SurfComplexityPen, int winsize,
			  int *dsi, int maxdisp, float scale, 
			  char *name, IplImage *left, IplImage *occ_left, char *disp_fn)
// main logic / get proposals from generator and fuse them with the current solution
{
	// get the initial solution
	Proposal *bestsolution = generator.GetProposal(0);

	while (1)
	{
		// get a new proposal from the generator
		// we pass the current solution because some proposal (e.g., refit) need it
		Proposal *proposal = generator.GetProposal(bestsolution);

		// if all proposals have been tested the generator returns 0
		// in this case we exit
		if (!proposal)
			break;

		// fuses bestsolution and proposal such that energy (1) is minimized
		Proposal *newbest = fuse2solutions (bestsolution, proposal,
											OccPen, SmoothPen, HOPen, segimages,
											SparsePen, SurfComplexityPen, winsize,
											name, left, occ_left, dsi, disp_fn);

		// cleanup
		delete proposal;
		delete bestsolution;

		// our current best solution is the result of the fusion move
		bestsolution = newbest;

	}

	// return the solution of lowest energy
	return *bestsolution;
}