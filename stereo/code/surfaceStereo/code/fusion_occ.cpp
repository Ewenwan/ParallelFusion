#include "fusion.h"

///////////////////////////////////////////////////////
//			Occlusion Term
#define MYINFINITY 1000
//#define INFINITY 0

int _imgW, _imgH;
float _scale;
int _maxdisp;
int _OccPen;
int *_dsi;

vector<int> *_oldlabels;

typedef struct
{
	CvPoint left_p;

	// used to identify
	int proposalid;
	int segid;

	float d;
	
	CvPoint right_p;

	int x_i;
} Entry;

vector<Entry> **Buffer = 0;

typedef struct
{
	CvPoint left_p;
	int x_i;
} PixelState;

// two pixels interact if one occludes the other (also see eq.(4))
typedef pair<PixelState, PixelState> Interaction; // Occluder / Occlussion Pixel

/////////////////////////////////////////////
//	identification of occlusion interactions
// also see data term in Woodfords second order stereo paper
int lessthanEntry (Entry &e1, Entry &e2)
{
	  if (e1.d < e2.d)
		  return 1;
	  if (e1.d > e2.d)
		  return 0;

	  if (e1.proposalid < e2.proposalid)
		  return 1;
	  if (e1.proposalid > e2.proposalid)
		  return 0;

	  if (e1.segid < e2.segid)
		  return 1;
	  if (e1.segid > e2.segid)
		  return 0;

	  if (e1.x_i < e2.x_i)
		  return 1;
	 
	  if (e1.x_i > e2.x_i)
		  return 0;

	  printf ("check the entry lessthan!");

	  return 1;
}

// buffer represent a warped right view
// pixel mapping to the same cell in the warped view have an occlusion interation
void AllocBuffer(Proposal *proposal)
{
	if (Buffer != 0)
	{
		for (int i = 0; i < _imgW * _imgH; i++)
			Buffer[i]->clear();
	}
	else
	{
		Buffer = (vector<Entry>**) malloc (_imgW * _imgH * sizeof (vector<Entry>*));
		for (int i = 0; i < _imgW * _imgH; i++)
			Buffer[i] = new vector<Entry>;
	}
}

void AddEntry (Entry &entry)
// adds an entry to the cell in the right image
// sort all pixels according to disparity to find which pixels is occluded and which is the occluder
{
	if (entry.right_p.x < 0 || entry.right_p.x >= _imgW)
		return;

	vector<Entry> *cell = Buffer[entry.right_p.y * _imgW + entry.right_p.x];

	vector<Entry>::iterator it;
	for (it = cell->begin(); it != cell->end(); it++)
	{
		if (lessthanEntry (*it, entry))
			break;
	}

	cell->insert(it, entry);
}

Entry GenerateEntry (SurfaceModel *sm, CvPoint left_p)
// constructs an entry. the important point is to generate the entry's x-coordinate in the right view
{
	Entry entry;

	entry.left_p = left_p;

	entry.proposalid = sm->proposalid;
	entry.segid = sm->segid;

	entry.d = sm->PointDisp (left_p, _scale);
	
	entry.right_p = left_p;
	entry.right_p.x -= (int) (entry.d + 0.5f);

	// x_i is filled externally

	return entry;
}

vector<Interaction> GetInteractions (Proposal *proposal1, Proposal *proposal2)
// list all possible pixel interations (occluder / occluded) between pixels of proposals 1 and 2
{
	// allocate
	AllocBuffer(proposal1);

	// all occluder/occluded - interaction that can occor for all combinations of proposals 1 and 2 
	vector<Interaction> interactions;

	// fill the buffer representing the right view
	for (int y = 0; y < _imgH; y++)
		for (int x = 0; x < _imgW; x++)
		{
			SurfaceModel *sm1 = proposal1->surfacemodels[y * _imgW + x];
			SurfaceModel *sm2 = proposal2->surfacemodels[y * _imgW + x];

			Entry entry0 = GenerateEntry (sm1, cvPoint (x,y));
			entry0.x_i = 0;
			AddEntry (entry0);

			Entry entry1 = GenerateEntry (sm2, cvPoint (x,y));
			entry1.x_i = 1;
			AddEntry (entry1);
		}

	// read the interactions
	// if a cell of the buffer contains more than 1 entry then the one of higher disparity
	// occludes all others of lower disparities
	for (int y = 0; y < _imgH; y++)
		for (int x = 0; x < _imgW; x++)
		{
			vector<Entry> *cell = Buffer[y * _imgW + x];

			if (cell->size() < 2)
				continue;

			for (int i = 0; i < cell->size() - 1; i++)
			{
				Entry occluder = cell->at(i);

				for (int j = i + 1; j < cell->size(); j++)
				{
					Entry occlusion = cell->at(j);

					// no interaction for identical surfaces
					// see paper (improved asymmetric occlusion model)
					if (occluder.proposalid == occlusion.proposalid &&
						occluder.segid == occlusion.segid)
						continue;

					// build occluder / occlusion pair with corresponding states
					Interaction interaction;

					interaction.first.left_p = occluder.left_p;
					interaction.first.x_i = occluder.x_i;

					interaction.second.left_p = occlusion.left_p;
					interaction.second.x_i = occlusion.x_i;
					
					interactions.push_back (interaction);
				}
			}
		}


	return interactions;
}

// QPBOI workaround
int hasSwappedMeaning (int idx)
{
	return _oldlabels->at(idx);
}

void AddDataTermWithOcc (kolmogorov::qpbo::QPBO<REAL> *q, Proposal *proposal1, Proposal *proposal2, int *dsi, int OccPen, vector<int> *oldlabels)
{
	_imgW = proposal1->imgW;
	_imgH = proposal1->imgH;
	_scale = proposal1->scale;
	_maxdisp = proposal1->maxdisp;
	_OccPen = OccPen;
	_dsi = dsi;
	_oldlabels = oldlabels;

	// We add a node for each pixel and two nodes representing its occlusion state (occ0 and occ1)
	// data / occlusion costs are implemented as pairwise potentials between pixel and occlusion nodes
	// See Woodford's paper
	q->AddNode (3 * _imgW * _imgH);

	// offset for simple access
	int o0_shift = _imgW * _imgH;
	int o1_shift = 2 * _imgW * _imgH;

	// add photo consistency
	for (int y = 0; y < _imgH; y++)
		for (int x = 0; x < _imgW; x++)
		{
			int d_idx = y * _imgW + x;

			// compute matching costs for both proposal
			int c0 = MatchingCosts (cvPoint (x,y), *proposal1, _dsi, _OccPen);
			int c1 = MatchingCosts (cvPoint (x,y), *proposal2, _dsi, _OccPen);

			// add the pairwise links
			int o0_idx = d_idx + o0_shift;
			int o1_idx = d_idx + o1_shift;

			// QPBOI workaround
			int o0_swap = hasSwappedMeaning (o0_idx);
			int o1_swap = hasSwappedMeaning (o1_idx);

			if (o0_swap == 0)
			{
				// we give matching costs if pixel is visible and occlusion penalty otherwise
				q->AddPairwiseTerm (d_idx, o0_idx,
					c0, _OccPen, 0, 0);
			}
			else
			{
				q->AddPairwiseTerm (d_idx, o0_idx,
					_OccPen, c0, 0, 0);
			}

			if (o1_swap == 0)
			{
				q->AddPairwiseTerm (d_idx, o1_idx,
					0, 0, c1, _OccPen);
			}
			else
			{
				q->AddPairwiseTerm (d_idx, o1_idx,
					0, 0, _OccPen, c1);
			}
		}

	// get all occlusion interactions (if two pixels map to the same cell, the occlusion node of the pixel 
	// with lower disparity has to be in state occluded. Otherwise give infinite costs.
	vector<Interaction> interactions = GetInteractions (proposal1, proposal2);

	for (int i = 0; i < interactions.size(); i++)
	{
		PixelState occluder = interactions[i].first;
		PixelState occlusion = interactions[i].second;

		int occluder_idx = occluder.left_p.y * _imgW + occluder.left_p.x;
		
		int o_offest = occlusion.left_p.y * _imgW + occlusion.left_p.x;
		int occlusion_idx;

		// implement infinite edge costs to prevent situation described above
		if (occluder.x_i == 0)
		{
			if (occlusion.x_i == 0)
				occlusion_idx = o_offest + o0_shift;
			else
				occlusion_idx = o_offest + o1_shift;

			int swapped = hasSwappedMeaning (occlusion_idx);
			if (!swapped)
			{
				q->AddPairwiseTerm (occluder_idx, occlusion_idx,
					MYINFINITY, 0, 0, 0);
			}
			else
			{
				q->AddPairwiseTerm (occluder_idx, occlusion_idx,
					0, MYINFINITY, 0, 0);
			}

		}
		if (occluder.x_i == 1)
		{
			if (occlusion.x_i == 0)
				occlusion_idx = o_offest + o0_shift;
			else
				occlusion_idx = o_offest + o1_shift;

			int swapped = hasSwappedMeaning (occlusion_idx);
			if (!swapped)
			{
				q->AddPairwiseTerm (occluder_idx, occlusion_idx,
					0, 0, MYINFINITY, 0);
			}
			else
			{
				q->AddPairwiseTerm (occluder_idx, occlusion_idx,
					0, 0, 0, MYINFINITY);
			}
		}
	}
}

// manually compute data costs (used for double checking)
int Evaluate_DataWithOcc (kolmogorov::qpbo::QPBO<REAL> *q, Proposal *proposal1, Proposal *proposal2)
{
	int e = 0;

	for (int y = 0; y < _imgH; y++)
		for (int x = 0; x < _imgW; x++)
		{
			int idx = y * _imgW + x;

			int x_i = q->GetLabel (idx);

			int occ_idx;

			Proposal *curprop;
			if (x_i == 0)
			{
				occ_idx = idx + _imgW * _imgH;
				curprop = proposal1;
			}
			else if (x_i == 1)
			{
				occ_idx = idx + 2 * _imgW * _imgH;
				curprop = proposal2;
			}

			int occluded = q->GetLabel (occ_idx);
			if (hasSwappedMeaning (occ_idx))
				occluded = (!occluded);

			if (!occluded)
				e += MatchingCosts (cvPoint (x,y), *curprop, _dsi, _OccPen);
			else 
				e += _OccPen;
		}
	
	return e;
}

void Getoldlabels (kolmogorov::qpbo::QPBO<REAL> *q, vector<int> *&oldlabels)
{
	vector <int> *newlabels = new vector<int>;

	for (int i = 0; i < 3 * _imgW * _imgH; i++)
		newlabels->push_back(0);

	for (int i = 0; i < _imgW * _imgH; i++)
	{
		// get current occlusion state
		int x_i = q->GetLabel (i);

		int occ_idx;
		if (x_i == 0)
			occ_idx = i + _imgW * _imgH;
		if (x_i == 1)
			occ_idx = i + 2 * _imgW * _imgH;

		int occluded = q->GetLabel(occ_idx);
		if (hasSwappedMeaning(occ_idx))
			occluded = (!occluded);

		// set o0 labels according to current occlusion state
		newlabels->at (i + _imgW * _imgH) = occluded;
		newlabels->at (i + 2 * _imgW * _imgH) = 1;
	}

	delete oldlabels;
	oldlabels = newlabels;
}

// plots disparity map with occlusions in red
void ShowOcclusionGraph (kolmogorov::qpbo::QPBO<REAL> *q, Proposal *proposal1, Proposal *proposal2, float scale, char *fn)
{
	int imgW = proposal1->imgW;
	int imgH = proposal1->imgH;

	IplImage *disp = cvCreateImage(cvSize(imgW, imgH), IPL_DEPTH_8U, 3);
	IplImage *occ = cvCreateImage(cvSize(imgW, imgH), IPL_DEPTH_8U, 3);

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			int disp_node = q->GetLabel(y * imgW + x);

			Proposal *solution = 0;
			if (disp_node == 0)
				solution = proposal1;
			else if (disp_node == 1)
				solution = proposal2;
			else
			{
				disp->imageData[y * disp->widthStep + 3 * x + 0] = 0;
				disp->imageData[y * disp->widthStep + 3 * x + 1] = 255;
				disp->imageData[y * disp->widthStep + 3 * x + 2] = 0;

				occ->imageData[y * disp->widthStep + 3 * x + 0] = 0;
				occ->imageData[y * disp->widthStep + 3 * x + 1] = 255;
				occ->imageData[y * disp->widthStep + 3 * x + 2] = 0;

				continue;
			}

			float d = solution->surfacemodels[y * imgW + x]->PointDisp(cvPoint(x, y), scale);
			d *= scale;
			int int_d = min ((int) (d + 0.5f), 255);

			for (int c = 0; c < 3; c++)
				disp->imageData[y * disp->widthStep + 3 * x + c] = (uchar) int_d;

			int occidx;

			if (disp_node == 0)
				occidx = y * imgW + x + imgW * imgH;
			else
				occidx = y * imgW + x + 2 * imgW * imgH;

			int occluded = q->GetLabel(occidx);

			if (hasSwappedMeaning(occidx))
				occluded = (!occluded);

			if (occluded)
			{
				occ->imageData[y * disp->widthStep + 3 * x + 0] = 0;
				occ->imageData[y * disp->widthStep + 3 * x + 1] = 0;
				occ->imageData[y * disp->widthStep + 3 * x + 2] = 255;
			}
			else
			{
				for (int c = 0; c < 3; c++)
					occ->imageData[y * disp->widthStep + 3 * x + c] = 
						disp->imageData[y * disp->widthStep + 3 * x + c];
			}
		}


	#ifdef SHOWIMAGES
		cvvNamedWindow ("graph occ", CV_WINDOW_AUTOSIZE);
		cvShowImage ("graph occ", occ);
	#endif

	cvSaveImage (fn, occ);

	cvReleaseImage (&disp);
	cvReleaseImage (&occ);
}