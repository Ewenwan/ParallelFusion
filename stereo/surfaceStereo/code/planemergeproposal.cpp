#include "proposals.h"

inline float CuttingDistance (CvPoint3D32f &Centroid, CvPoint3D32f &normVec, Plane *plane)
{
	float k = (- plane->A * Centroid.x - plane->B * Centroid.y + Centroid.z - plane->C) /
			  (  plane->A * normVec.x  + plane->B * normVec.y  - normVec.z);

	float Vec[3];
	Vec[0] = k * normVec.x;
	Vec[1] = k * normVec.y;
	Vec[2] = k * normVec.z;

	float length = sqrt (Vec[0] * Vec[0] + Vec[1] * Vec[1] + Vec[2] * Vec[2]);

	return length;
}

float PlaneSimilarity (Plane *plane_A, CvPoint3D32f Centroid_A, Plane *plane_B, CvPoint3D32f Centroid_B)
{
	CvPoint3D32f normVec;
	
	// cut normal vec of Plane_A with plane Plane_B
	normVec.x = plane_A->A;
	normVec.y = plane_A->B;
	normVec.z = (float) -1.0;

	float dist = CuttingDistance (Centroid_A, normVec, plane_B);

	// cut normal vec of Plane_B with plane Plane_A
	normVec.x = plane_B->A;
	normVec.y = plane_B->B;
	normVec.z = (float) -1.0;

	dist += CuttingDistance (Centroid_B, normVec, plane_A);

	return dist;
}

CvPoint3D32f CenterOfGravity(Segment &seg, Plane *plane, float scale)
{
	CvPoint3D32f p;

	int i, n, sumx, sumy;
	float x, y;

	sumx = sumy = 0.0;
	n = 0;
	for (i = 0; i < seg.segpoints.size(); i++)
	{
		sumx += seg.segpoints[i].x;
		sumy += seg.segpoints[i].y;
		n++;
	}

	p.x = sumx / (float) n;
	p.y = sumy / (float) n;

	CvPoint p_int;
	p_int.x = (int) (p.x + 0.5f);
	p_int.y = (int) (p.y + 0.5f);
	p.z = plane->PointDisp(p_int, scale);

	return p;
}

void id2col (int id, char *colour)
{
	colour[0] = id * 543 % 255;
	colour[1] = id * 65764 % 255;
	colour[2] = id * 4321431 % 255;
}

typedef struct PlanarSegment
{
	Segment segment;
	Plane *plane;
	CvPoint3D32f CenterOfGravity;

	set<int> ids;
};

typedef struct MergePair
{
	int id1;
	int id2;
};

void Proposal::gernerate_planemerge_proposal (Proposal *src_proposal, float planesim_threshold)
{
	IplImage *ids_img = src_proposal->plot_surface_ids ();

	Segments segments;
	segments.InitFromSegmentedImage (ids_img);
	cvReleaseImage (&ids_img);

	// generate planar segments
	vector<PlanarSegment> planarsegments;
	for (int i = 0; i < segments.size(); i++)
	{
		Segment seg = segments.at(i);

		if (seg.segpoints.size() <= 0)
			continue;

		CvPoint p = seg.segpoints[0];
		SurfaceModel *sm = src_proposal->surfacemodels[p.y * imgW + p.x];

		if (sm->which_type() != IS_PLANE && sm->which_type() != IS_CONSTANT_SURFACE)
		{
			printf ("plane merger used on non-planar surface model\n");
			continue;
		}

		PlanarSegment pseg;
		pseg.ids.insert(i);
		pseg.segment = seg;
		pseg.plane = (Plane *) sm;
		pseg.CenterOfGravity = CenterOfGravity(pseg.segment, pseg.plane, scale);

		planarsegments.push_back (pseg);
	}

	vector<MergePair> mergepairs;

	// find the segments that shall be merged
	for (int i = 0; i < planarsegments.size(); i++)
		for (int j = i + 1; j < planarsegments.size(); j++)
		{
			float planesim = PlaneSimilarity (planarsegments[i].plane, planarsegments[i].CenterOfGravity, 
											  planarsegments[j].plane, planarsegments[j].CenterOfGravity);

			if (planesim <= planesim_threshold)
			{
				MergePair mp;
				mp.id1 = *(planarsegments[i].ids.begin());
				mp.id2 = *(planarsegments[j].ids.begin());

				mergepairs.push_back(mp);
			}
		}

	vector<PlanarSegment> mergedsegs;
	vector<PlanarSegment>::iterator it1, it2;

	for (int i = 0; i < planarsegments.size(); i++)
		mergedsegs.push_back(planarsegments.at(i));

	for (int i = 0; i < mergepairs.size(); i++)
	{
		int id1 = mergepairs[i].id1;
		int id2 = mergepairs[i].id2;

		set<int>::iterator setit;

		for (it1 = mergedsegs.begin(); it1 != mergedsegs.end(); it1++)
		{
			setit = it1->ids.find(id1);
			if (setit != it1->ids.end())
				break;
		}
		for (it2 = mergedsegs.begin(); it2 != mergedsegs.end(); it2++)
		{
			setit = it2->ids.find(id2);
			if (setit != it2->ids.end())
				break;
		}

		if (it1 == it2)
			continue;

		// merge
		vector<CvPoint> seg2points = it2->segment.segpoints;
		for (int i = 0; i < seg2points.size(); i++)
			it1->segment.segpoints.push_back (seg2points[i]);
		
		set<int> seg2ids = it2->ids;
		for (setit = seg2ids.begin(); setit != seg2ids.end(); setit++)
			it1->ids.insert (*setit);

		mergedsegs.erase (it2);
	}

	int numpts = 0;
	IplImage *mergeimg = cvCreateImage (cvSize(imgW, imgH), IPL_DEPTH_8U, 3);
	cvZero (mergeimg);

	for (int i = 0; i < mergedsegs.size(); i++)
	{
		char col[3];
		id2col (i, col);

		vector<CvPoint> segpoints = mergedsegs[i].segment.segpoints;
		for (int j = 0; j < segpoints.size(); j++)
		{
			numpts++;

			CvPoint p = segpoints[j];
			for (int c = 0; c < 3; c++)
				mergeimg->imageData [p.y * mergeimg->widthStep + 3 * p.x + c] = col[c];
		}
	}

	if (numpts != imgW * imgH)
		printf ("error\n");

	printf ("before merge: %d after: %d\n", segments.size(), mergedsegs.size());

/*	cvvNamedWindow ("merged", CV_WINDOW_AUTOSIZE);
	cvMoveWindow ("merged", 0, 0);
	cvShowImage ("merged", mergeimg);
	cvWaitKey(0);*/

	IplImage *disp = src_proposal->plot_proposal(scale);
	gernerate_proposal_via_disp_and_segmentation (disp, mergeimg, 0, 1.0);

	cvReleaseImage (&mergeimg);
	cvReleaseImage (&disp);
}