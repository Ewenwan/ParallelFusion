#include "proposals.h"

vector<CvPoint> Dilation (vector<CvPoint> points, int imgW, int imgH)
{
	IplImage *binmask = cvCreateImage (cvSize (imgW, imgH), IPL_DEPTH_8U, 1);
	cvZero (binmask);

	for (int i = 0; i < points.size(); i++)
		binmask->imageData[points[i].y * binmask->widthStep + points[i].x] = (char) 255;

/*	cvvNamedWindow ("binmask", CV_WINDOW_AUTOSIZE);
	cvMoveWindow ("binmask", 0, 0);
	cvShowImage ("binmask", binmask);*/

	cvDilate (binmask, binmask, NULL, 2);

	vector<CvPoint> dilpoints;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
			if (binmask->imageData [y * binmask->widthStep + x] == (char) 255)
				dilpoints.push_back (cvPoint (x,y));

/*	cvvNamedWindow ("dil binmask", CV_WINDOW_AUTOSIZE);
	cvMoveWindow ("dil binmask", imgW, 0);
	cvShowImage ("dil binmask", binmask);
	cvWaitKey(0);*/

	cvReleaseImage (&binmask);

	return dilpoints;
}

Proposal *segmentdilation (Proposal *cursolution)
{
	map<SurfaceModel*, vector<CvPoint>, lessthanSurfaceModel> segments;
	map<SurfaceModel*, vector<CvPoint>, lessthanSurfaceModel>::iterator mapit;

	int imgW = cursolution->imgW;
	int imgH = cursolution->imgH;

	int *newlabels = (int*) malloc (imgW * imgH * sizeof(int));
	for (int i = 0; i < imgW * imgH; i++)
		newlabels[i] = -1;

	// get lists of segments
	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			SurfaceModel *sm = cursolution->surfacemodels[y * imgW + x];

			segments[sm].push_back (cvPoint(x,y));
		}

	for (int it = 0; it < segments.size() * 1.5; it++)
	{
		int expand_idx = rand() % segments.size();

		mapit = segments.begin();
		for (int i = 0; i < expand_idx; i++)
			mapit++;

		vector<CvPoint> dilpoints = Dilation (mapit->second, imgW, imgH);

		for (int i = 0; i < dilpoints.size(); i++)
			newlabels[dilpoints[i].y * imgW + dilpoints[i].x] = expand_idx;
	}

	// fill empty pixels with randow surface
	int idx = rand() % segments.size();
	for (int i = 0; i < imgW * imgH; i++)
		if (newlabels[i] == -1)
			newlabels[i] = idx;

	Proposal *proposal = new Proposal (cursolution);

	vector<SurfaceModel*> oldmodels, newmodels;

	for (mapit = segments.begin(); mapit != segments.end(); mapit++)
	{
		oldmodels.push_back (mapit->first);
		newmodels.push_back (0);
	}

	for (int i = 0; i < imgW * imgH; i++)
	{
		int label = newlabels[i];

		if (newmodels[label] == 0)
		{
			SurfaceModel *sm = oldmodels[label]->Clone();
			newmodels[label] = sm;
		}

		proposal->surfacemodels[i] = newmodels[label];
	}

	free (newlabels);

	return proposal;
}