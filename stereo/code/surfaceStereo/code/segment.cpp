#include "segment.h"

Segments::Segments()
{
	// 8 connectivity
	connectivity.push_back (cvPoint(0,1)); // up
	connectivity.push_back (cvPoint(1,1)); // up right
	connectivity.push_back (cvPoint(1,0)); // right
	connectivity.push_back (cvPoint(1,-1)); // down right
	connectivity.push_back (cvPoint(0,-1)); // down
	connectivity.push_back (cvPoint(-1,-1)); // down left
	connectivity.push_back (cvPoint(-1,0)); // left
	connectivity.push_back (cvPoint(-1,1)); // up left
}

void char2int (IplImage *img_char, IplImage *img_int)
// convert 3 channel char image to 1 channel int image
{
	int *data = (int*) img_int->imageData;

	for (int y = 0; y < img_char->height; y++)
		for (int x = 0; x < img_char->width; x++)
		{
			int val = (int) (uchar) img_char->imageData [y * img_char->widthStep + 3 * x + 0];
			val +=    256 * (int) (uchar) img_char->imageData [y * img_char->widthStep + 3 * x + 1];
			val +=    256 * 256 * (int) (uchar) img_char->imageData [y * img_char->widthStep + 3 * x + 2];
	
			*data = val;
			data++;
		}
}

inline int PointInsideImage (CvPoint Point, int imgW, int imgH)
{
	if (Point.x >= 0 && Point.x < imgW &&
		Point.y >= 0 && Point.y < imgH)
			return 1;
	else
			return 0;
}

void Segments::ReadSegmentsFromLabelledImage (IplImage *labelled)
// read all segments from an integer image
{
	segments.clear();

	int imgW = labelled->width;
	int imgH = labelled->height;

	Segment segment;
	CvPoint *queue_begin, *queue_end;
	CvPoint current;
	int curlabel;

	int *labeldata = (int*) labelled->imageData;
	int *searched = (int*) malloc (imgW * imgH * sizeof (int));
	CvPoint *searchqueue = (CvPoint*) malloc (8 * imgW * imgH * sizeof (CvPoint));

	memset (searched, 0, imgW * imgH * sizeof (int));

	int segid = 0;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			// find first point that is not already member of a segment
			if (!searched[y * imgW + x])
			{
				*searchqueue = cvPoint (x,y);

				queue_begin = searchqueue;
				queue_end = queue_begin + 1;

				curlabel = labeldata [y * imgW + x];
				segment.segpoints.clear();

				// expand the points neighbours
				while (queue_begin != queue_end)
				{
					current = *queue_begin++;
					
					if (searched[current.y * imgW + current.x])
						continue;

					segment.segpoints.push_back (current);
					searched[current.y * imgW + current.x] = 1;

					for (unsigned int i = 0; i < connectivity.size(); i++)
					{
						CvPoint neighbour = cvPoint (current.x + connectivity[i].x,
													 current.y + connectivity[i].y);

						// check if neighbour is inside image bounds
						if (!PointInsideImage (neighbour, imgW, imgH))
							continue;
						// check if point has already been processed
						if (searched[neighbour.y * imgW + neighbour.x])
							continue;
						// check if point carrys the label of the current segment
						if (labeldata[neighbour.y * imgW + neighbour.x] != curlabel)
							continue;

						// if all conditions are meet put the point into the queue
						*queue_end++ = neighbour;
					}
				}
				segment.segid = segid++;
				segments.push_back (segment);
			}
		}

	free (searched);
	free (searchqueue);
}

void Segid_To_Colour (int segid, char *colour)
{
	colour[0] = segid * 543 % 255;
	colour[1] = segid * 65764 % 255;
	colour[2] = segid * 4321431 % 255;
}

void Segments::Plot (IplImage *dst)
{
	cvZero (dst);

	for (unsigned int i = 0; i < segments.size(); i++)
		for (unsigned int p = 0; p < segments[i].segpoints.size(); p++)
		{
			int id = segments[i].segid;
			CvPoint point = segments[i].segpoints[p];

			char colour[3];
			Segid_To_Colour (id, colour);

			for (int c = 0; c < 3; c++)
				dst->imageData[point.y * dst->widthStep + 3 * point.x + c] = colour[c];
		}
}

void UpdateConfigFile (char *path, float SpatialBandwidth, float RangeBandwidth, float MinimumRegionsArea)
{
	char fn[512] = {};
	sprintf (fn, "%s/source.txt", path);
	FILE * fp_in = fopen (fn,"r");

	sprintf (fn, "%s/config.txt", path);
	FILE * fp_out = fopen (fn,"w");

	int i = 1;
	while(!feof(fp_in))
	{
		char text[1024];
		fgets (text, sizeof (text), fp_in);
		//fscanf (fp_in, "%s\n", text);

		if (i == 6)
			fprintf (fp_out, "SpatialBandwidth = %d;\n", (int) SpatialBandwidth);
		else if (i == 7)
			fprintf (fp_out, "RangeBandwidth = %.1f;\n", RangeBandwidth);
		else if (i == 8)
			fprintf (fp_out, "MinimumRegionArea = %d;\n", (int) MinimumRegionsArea);
		else
			fprintf (fp_out, "%s", text);

		i++;
	}

	fclose (fp_in);
	fclose (fp_out);
}

IplImage *Segments::InitFromMeanShift (char *imgname, IplImage *img, float SpatialBandwidth, float RangeBandwidth, float MinimumRegionsArea)
{
	char fn[2048];
	sprintf (fn, "%s\\%s_SB_%.0f_RB_%.0f_MR_%d.ppm", EDISON_PATH, imgname, SpatialBandwidth, RangeBandwidth, MinimumRegionsArea);

	// check if segmentation already exists
	IplImage *labelled_char;
	labelled_char = cvLoadImage (fn);

	if (!labelled_char)
	{
		UpdateConfigFile (EDISON_PATH, SpatialBandwidth, RangeBandwidth, MinimumRegionsArea);

		char dummy[512] = {};

		sprintf (dummy, "%s/in.ppm", EDISON_PATH);


		IplImage *temp = cvCloneImage (img);
		cvSaveImage(dummy, temp);

		printf ("mean shift segmentation started\n");
		system(EDISON_EXE);
		printf ("mean shift segmentation finished\n");

		sprintf (dummy, "%s/out.ppm", EDISON_PATH);

		labelled_char = cvLoadImage (dummy);

		cvSaveImage (fn, labelled_char);
	}

	// convert 3 channel char image to 1 channel int image
	IplImage *labelled = cvCreateImage (cvSize (labelled_char->width, labelled_char->height), IPL_DEPTH_32S, 1);
	char2int (labelled_char, labelled);

	ReadSegmentsFromLabelledImage (labelled);
	cvReleaseImage (&labelled);

	Plot (labelled_char);

	printf ("num segments %d\n", segments.size());

	return labelled_char;
}

void Segments::InitFromSegmentedImage (IplImage *segimage)
{
	// convert 3 channel char image to 1 channel int image
	IplImage *labelled = cvCreateImage (cvSize (segimage->width, segimage->height), IPL_DEPTH_32S, 1);
	char2int (segimage, labelled);

	ReadSegmentsFromLabelledImage (labelled);
	cvReleaseImage (&labelled);

	printf ("num segments %d\n", segments.size());
}