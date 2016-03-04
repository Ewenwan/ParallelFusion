#include "kmeans.h"

typedef struct ClusterCenter
{
	float d;
	float x;
	float y;

	vector<CvPoint> members;
};

vector<ClusterCenter> InitRandomCenters (IplImage *disp, int NumberOfClusters)
{
	int imgW = disp->width;
	int imgH = disp->height;

	vector<ClusterCenter> centers;

	for (int i = 0; i < NumberOfClusters; i++)
	{
		ClusterCenter center;
		int x = rand() % imgW;
		int y = rand() % imgH;

		center.d = (float) (uchar) disp->imageData[y * disp->widthStep + x];
		center.d /= 255.f;

		center.x = (float) x / (float) imgW;
		center.y = (float) y / (float) imgH;

		centers.push_back (center);
	}

	return centers;
}

inline float DistanceToCenter (float x, float y, float d, ClusterCenter &center)
{
	float xdif = x - center.x;
	xdif *= xdif;

	float ydif = y - center.y;
	ydif *= ydif;

	float ddif = d - center.d;
	ddif *= ddif;

	float sum = xdif + ydif + 200 * ddif;

	return sum;

}

void AssignMembers (IplImage *disp, vector<ClusterCenter> &centers)
{
	int imgW = disp->width;
	int imgH = disp->height;

	// assign members
	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			float norm_x = (float) x / (float) imgW;
			float norm_y = (float) y / (float) imgH;

			float norm_d = (float) (uchar) disp->imageData[y * disp->widthStep + x];
			norm_d /= 255.f;
			
			int bestcluster = 0;
			float bestdistance = 99999.f;

			for (int i = 0; i < centers.size(); i++)
			{
				float curdistance = DistanceToCenter (norm_x, norm_y, norm_d, centers[i]);
				if (curdistance < bestdistance)
				{
					bestdistance = curdistance;
					bestcluster = i;
				}
			}

			centers[bestcluster].members.push_back (cvPoint(x,y));
		}
}

void UpdateCenters (IplImage *disp, vector<ClusterCenter> &centers)
{
	int imgW = disp->width;
	int imgH = disp->height;

	for (int i = 0; i < centers.size(); i++)
	{
		float sum_x = 0.f;
		float sum_y = 0.f;
		float sum_d = 0.f;

		vector<CvPoint> members = centers[i].members;
	
		for (int j = 0; j < members.size(); j++)
		{
			sum_x += (float) members[j].x / (float) imgW;
			sum_y += (float) members[j].y / (float) imgH;

			float d = (float) (uchar) disp->imageData [members[j].y * disp->widthStep + members[j].x];
			d /= 255.f;
			sum_d += d;
		}

		centers[i].x = sum_x / (float) members.size();
		centers[i].y = sum_y / (float) members.size();
		centers[i].d = sum_d / (float) members.size();
		centers[i].members.clear();
	}
}

void GetColor (int id, char *colour)
{
	colour[0] = id * 543 % 255;
	colour[1] = id * 65764 % 255;
	colour[2] = id * 4321431 % 255;
}

IplImage *PlotSegmentation (IplImage *disp, vector<ClusterCenter> &centers)
{
	int imgW = disp->width;
	int imgH = disp->height;

	IplImage *segmented = cvCreateImage (cvSize (imgW, imgH), IPL_DEPTH_8U, 3);
	cvZero (segmented);

	for (int i = 0; i < centers.size(); i++)
	{
		vector<CvPoint> members = centers[i].members;
		char col[3];
		GetColor (i, col);

		for (int j = 0; j < members.size(); j++)
		{
			CvPoint p = members[j];
			for (int c = 0; c < 3; c++)
				segmented->imageData[p.y * segmented->widthStep + 3 * p.x + c] = col[c];
		}
	}

	return segmented;
}

IplImage *kmeans_segmentation (IplImage *disp, int NumberOfClusters)
{
	int imgW = disp->width;
	int imgH = disp->height;

	IplImage *segmented;

//	for (int j = 0; j < 30; j++)
	{

		vector<ClusterCenter> centers = InitRandomCenters (disp, NumberOfClusters);

		int it = 0;
		int maxit = 10;

		while (1)
		{
			AssignMembers (disp, centers);

			if (it >= maxit)
				break;

			UpdateCenters (disp, centers);

			it++;
		}

		segmented = PlotSegmentation (disp, centers);
		/*cvvNamedWindow ("segmented", CV_WINDOW_AUTOSIZE);
		cvMoveWindow ("segmented", 0, 0);
		cvShowImage ("segmented", segmented);*/
		//cvWaitKey(0);

	}

	return segmented;
}