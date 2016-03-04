#include "warp.h"

IplImage *Unscale_Image (IplImage *src, float scale)
{
	int imgW = src->width;
	int imgH = src->height;

	IplImage *unscaled = cvCloneImage (src);
	cvZero (unscaled);

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			float d_f = (float) (uchar) src->imageData[y * src->widthStep + x];
			int d = (int) (d_f / scale + 0.5f);
			unscaled->imageData[y * unscaled->widthStep + x] = (uchar) d;
		}

	return unscaled;
}

IplImage *Occ_Mask_Via_Disp_Map (IplImage *scaled_disp, float scale)
{
	IplImage *disp = Unscale_Image (scaled_disp, scale);

	int imgW = disp->width;
	int imgH = disp->height;

	IplImage *occmask = cvCloneImage (disp);
	cvZero (occmask);

	// warp to derive occlusions of second view
	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			int d = (int) (uchar) disp->imageData[y * disp->widthStep + x];

			if (x - d < 0) 
				continue;

			occmask->imageData[y * occmask->widthStep + x - d] = (uchar) 255;
		}

/*	// remove isolated occlusions
	for (int y = 0; y < imgH; y++)
		for (int x = 1; x < imgW - 1; x++)
		{
			int prevocc = (int) (uchar) occmask->imageData[y * occmask->widthStep + x - 1];
			int curocc = (int) (uchar) occmask->imageData[y * occmask->widthStep + x];
			int nextocc = (int) (uchar) occmask->imageData[y * occmask->widthStep + x + 1];

			if (prevocc == 255 && nextocc == 255 && curocc == 0)
				occmask->imageData[y * occmask->widthStep + x] = (uchar) 255;

			if (prevocc == 0 && nextocc == 0 && curocc == 255)
				occmask->imageData[y * occmask->widthStep + x] = (uchar) 0;
		}*/

	cvReleaseImage (&disp);

/*	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			occmask->imageData[y * occmask->widthStep + x] = (uchar) 255;
		}*/


	return occmask;
}