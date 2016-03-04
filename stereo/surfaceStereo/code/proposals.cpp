#include "proposals.h"

Proposal::Proposal (IplImage *_left, IplImage *_right, int _maxdisp, float _scale, int _proposalid)
{
	left = _left;
	right = _right;
	maxdisp = _maxdisp;
	scale = _scale;
	proposalid = _proposalid;

	imgW = left->width;
	imgH = left->height;

	surfacemodels.resize (imgW * imgH);
}

Proposal::Proposal (Proposal *p)
{
	left = p->left;
	right = p->right;
	maxdisp = p->maxdisp;
	scale = p->scale;
	proposalid = p->proposalid;

	imgW = left->width;
	imgH = left->height;

	surfacemodels.resize (imgW * imgH);
}

Proposal::~Proposal()
{
	set<SurfaceModel*,lessthanSurfaceModel> surfset;

	for (int i = 0; i < imgW * imgH; i++)
	{
		surfset.insert (surfacemodels[i]);
	}

	set<SurfaceModel*,lessthanSurfaceModel>::iterator setit;

	for (setit = surfset.begin(); setit != surfset.end(); setit++)
		delete *setit;
}

int Proposal::get_surface_cnt()
{
	set<SurfaceModel*,lessthanSurfaceModel> surfset;

	for (int i = 0; i < imgW * imgH; i++)
		surfset.insert (surfacemodels[i]);

	return surfset.size();
}

int Proposal::get_spline_cnt()
{
	set<SurfaceModel*,lessthanSurfaceModel> surfset;

	for (int i = 0; i < imgW * imgH; i++)
		surfset.insert (surfacemodels[i]);

	int spline_cnt = 0;

	set<SurfaceModel*,lessthanSurfaceModel>::iterator setit;

	for (setit = surfset.begin(); setit != surfset.end(); setit++)
	{
		if ((*setit)->which_type() == IS_SPLINE)
			spline_cnt++;
	}

	return spline_cnt;
}

void Proposal::gernerate_meanshift_proposal (char *imgname, float MS_SpatialBandwidth, float MS_RangeBandwidth, float MS_MinimumRegionsArea, int ST_P1, int ST_P2, int splinenumber, float plane_threshold)
{
	surfacemodels.clear();
	surfacemodels.resize (imgW * imgH);

	Segments segments;
	IplImage *segimg = segments.InitFromMeanShift (imgname, left, MS_SpatialBandwidth, MS_RangeBandwidth, MS_MinimumRegionsArea);

	//cvvNamedWindow ("Segmentation", CV_WINDOW_AUTOSIZE);
	//cvMoveWindow ("Segmentation", 0, 0);
	//cvShowImage ("Segmentation", segimg);

	IplImage *disp = ExternDisparityMap (imgname, left, right, maxdisp, scale, ST_P1, ST_P2);

	//cvvNamedWindow ("SimpleTree", CV_WINDOW_AUTOSIZE);
	//cvMoveWindow ("SimpleTree", left->width, 0);
	//cvShowImage ("SimpleTree", disp);
	//cvWaitKey(10);

	gernerate_proposal_via_disp_and_segmentation (disp, segimg, splinenumber, 1.0);

	cvReleaseImage (&segimg);
	cvReleaseImage (&disp);
}

void Proposal::gernerate_proposal_via_disp_and_segmentation (IplImage *disp, IplImage *segmentation, int splinenumber, float plane_threshold)
{
	surfacemodels.clear();
	surfacemodels.resize (imgW * imgH);

	Segments segments;
	segments.InitFromSegmentedImage(segmentation);

/*	cvvNamedWindow ("disp", CV_WINDOW_AUTOSIZE);
	cvMoveWindow ("disp", left->width, 0);
	cvShowImage ("disp", disp);

	cvWaitKey(10);*/

	for (int i = 0; i < segments.size(); i++)
	{
		if (splinenumber == 0)
		{
			FitPlaneToSegment_ROBUST (segments.segments[i], 1, disp, scale, plane_threshold);
			segments.segments[i].surfacemodel->proposalid = proposalid;

			Plane *plane = new Plane;
			*plane = *(Plane*) (segments.segments[i].surfacemodel);

			for (int j = 0; j < segments.segments[i].segpoints.size(); j++)
			{
				CvPoint point = segments.segments[i].segpoints[j];
				surfacemodels [point.y * imgW + point.x] = (SurfaceModel*) plane;
			}
		}
		else
		{
			FitSplineToSegment (segments.segments[i], disp, scale, splinenumber);
			segments.segments[i].surfacemodel->proposalid = proposalid;
			segments.segments[i].surfacemodel->segid = i;

			BSpline *spline = new BSpline ();
			*spline = *(BSpline*) (segments.segments[i].surfacemodel);

			for (int j = 0; j < segments.segments[i].segpoints.size(); j++)
			{
				CvPoint point = segments.segments[i].segpoints[j];
				surfacemodels [point.y * imgW + point.x] = (SurfaceModel*) spline;
			}
		}		
	}
}

void Proposal::Merge (Proposal *proposal1, Proposal *proposal2, vector<int> binlabels)
{
	set<SurfaceModel*, lessthanSurfaceModel> present;
	set<SurfaceModel*, lessthanSurfaceModel>::iterator setit;

	for (int i = 0; i < imgW * imgH; i++)
	{
		if (binlabels[i] == 0)
			surfacemodels[i] = proposal1->surfacemodels[i];
		else
			surfacemodels[i] = proposal2->surfacemodels[i];

		setit = present.find (surfacemodels[i]);

		if (setit == present.end())
		{
			SurfaceModel *newsm = surfacemodels[i]->Clone();
			surfacemodels[i] = newsm;
			present.insert (newsm);
		}
		else
			surfacemodels[i] = *setit;
	}
}

void Proposal::gernerate_constant_proposal (float A, float B, float C)
{
	surfacemodels.clear();
	surfacemodels.resize (imgW * imgH);

	Plane *plane = new Plane;
	plane->A = A;
	plane->B = B;
	plane->C = C;
	plane->proposalid = proposalid;
	plane->segid = 1;

	for (int i = 0; i < imgW * imgH; i++)
		surfacemodels [i] = (SurfaceModel*) plane;
}

IplImage *Proposal::plot_proposal (float scale)
{
	IplImage *disp = cvCreateImage(cvSize(imgW, imgH), IPL_DEPTH_8U, 1);
	cvZero(disp);

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			float d = surfacemodels[y * imgW + x]->PointDisp(cvPoint (x, y), scale);

			int d_int = (int) (min (d * scale, 255.f) + 0.5f);

			disp->imageData [y * disp->widthStep + x] = (uchar) d_int;
		}

	return disp;
}

IplImage *Proposal::plot_height_lines (IplImage *disp)
{
	IplImage *height_lines = cvCloneImage (disp);

	for (int y = 0; y < imgH - 1; y++)
		for (int x = 0; x < imgW - 1; x++)
		{
			height_lines->imageData [y * height_lines->widthStep + x] = (uchar) 255;

			int curd = (int) (uchar) disp->imageData[y * disp->widthStep + x];
			int rightd = (int) (uchar) disp->imageData[y * disp->widthStep + x + 1];
			int lowd = (int) (uchar) disp->imageData[(y + 1) * disp->widthStep + x + 1];

			if (curd != rightd || curd != lowd)
				height_lines->imageData [y * height_lines->widthStep + x] = (uchar) 0;
		}

	return height_lines;
}

void Ids_To_Colour (int p_id, int s_id, char *colour)
{
	colour[0] = (s_id * 543 +		p_id * 291) % 255;
	colour[1] = (s_id * 65764 +		p_id * 9763) % 255;
	colour[2] = (s_id * 4321431 +	p_id * 433) % 255;
}

IplImage *Proposal::plot_surface_ids ()
{
	IplImage *ids = cvCreateImage(cvSize(imgW, imgH), IPL_DEPTH_8U, 3);
	cvZero(ids);

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			// red color for invalid surfaces
			if (!surfacemodels[y * imgW + x]->is_valid())
			{
				ids->imageData[y * ids->widthStep + 3 * x + 0] = 0;
				ids->imageData[y * ids->widthStep + 3 * x + 1] = 0;
				ids->imageData[y * ids->widthStep + 3 * x + 2] = 255;
				continue;
			}

			int p_id = surfacemodels[y * imgW + x]->proposalid; 
			int s_id = surfacemodels[y * imgW + x]->segid;

			char colour[3];
			Ids_To_Colour (p_id, s_id, colour);

			for (int c = 0; c < 3; c++)
				ids->imageData[y * ids->widthStep + 3 * x + c] = colour[c];

			// gray if constant proposal
			if (surfacemodels[y * imgW + x]->which_type() == IS_CONSTANT_SURFACE || surfacemodels[y * imgW + x]->which_type() == IS_PLANE)
			{
				int grey = 0;
				for (int c = 0; c < 3; c++)
					grey += (int) colour[c];
				grey /= 3;
				for (int c = 0; c < 3; c++)
					ids->imageData[y * ids->widthStep + 3 * x + c] = grey;
			}
		}

	return ids;
}

IplImage *Proposal::plot_spline_order ()
{
	IplImage *orders = cvCreateImage(cvSize(imgW, imgH), IPL_DEPTH_8U, 1);
	cvZero(orders);

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			int splineorder = surfacemodels[y * imgW + x]->splineorder(); 

			int color = splineorder * 122;

			orders->imageData[y * orders->widthStep +x] = (uchar) color;
		}

	return orders;
}

IplImage *Median_Disparity_Map (vector<IplImage*> disp_maps)
{
	IplImage *median_disp = cvCloneImage (disp_maps[0]);
	cvZero(median_disp);

	int imgW = median_disp->width;
	int imgH = median_disp->height;

	int med_idx = disp_maps.size() / 2;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			vector<int> vals;
			for (int i = 0; i < disp_maps.size(); i++)
			{
				IplImage *cur_map = disp_maps[i];
				int d = (int) (uchar) cur_map->imageData [y * cur_map->widthStep + x];
				vals.push_back (d);
			}
			sort(vals.begin(), vals.end());
			median_disp->imageData[y * median_disp->widthStep + x] = vals[med_idx];
		}

	return median_disp;
}

void Proposal::Save (char *fn)
{
	FILE *fp = fopen (fn, "w");

	set<SurfaceModel*, lessthanSurfaceModel> present;
	set<SurfaceModel*, lessthanSurfaceModel>::iterator setit;

	for (int i = 0; i < imgW * imgH; i++)
	{
		setit = present.find (surfacemodels[i]);

		if (setit == present.end())
			present.insert (surfacemodels[i]);
	}

	for (setit = present.begin(); setit != present.end(); setit++)
	{
		(*setit)->Save (fp);
	}

	fprintf (fp, "------------\n");

	for (int i = 0; i < imgW * imgH; i++)
	{
		SurfaceModel *cur = surfacemodels[i];
		fprintf (fp, "[%d,%d]\n", cur->proposalid, cur->segid);
	}

	fclose (fp);
}

void Proposal::Load (char *fn)
{
	FILE *fp = fopen (fn, "r");

	char *data = (char*) malloc (2048 * sizeof (char));

	while (1)
	{
		fscanf (fp, "%s\n", data);
		if (strcmp (data, "------------"))
			break;
	}

	set<SurfaceModel*, lessthanSurfaceModel> present;


	free (data);
}