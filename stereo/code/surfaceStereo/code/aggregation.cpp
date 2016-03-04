#include "aggregation.h"

int *aggregate (int *dsi, int imgW, int imgH, int maxdisp, int winsize)
{
	printf ("preforming aggregation\n");

	int winsize_div2 = winsize / 2;

	int i, x, y, d, sum;

	// DSI structure
	int *aggregated = (int*) malloc(imgH * imgW * (maxdisp + 1) * sizeof(int));
	// init 
	memset (aggregated, 0, imgH * imgW * (maxdisp + 1) * sizeof(int));

	// array that holds the pixel dissimilarity, 
	// we extend the borders to allow for an efficient incremental computation
	int *pixelDis = (int*) malloc ((imgW + winsize - 1) * sizeof(int));
	memset (pixelDis, 0, (imgW + winsize - 1) * sizeof(int));
	pixelDis += winsize_div2;

	// 2 dimensional array for storing the xRows, also extended for efficiency
	int *xRows = (int*) malloc ((imgH + winsize - 1) * imgW * sizeof(int));
	memset (xRows, 0, (imgH + winsize - 1) * imgW * sizeof(int));

	xRows += winsize_div2 * imgW;

	int *SliceInD = (int*) malloc (imgW * imgH * sizeof(int));
	int *SlicePtr;

	int *first, *next, *xRowPtr;

	for (d = 0; d <= maxdisp; d++)
	{
		for (y = 0; y < imgH; y++)
		{
			int *tmp = pixelDis;
			// copy the values from the DSI
			for (x = 0; x < imgW; x++)
			{
				*tmp = dsi[y * imgW * (maxdisp + 1) + x * (maxdisp + 1) + d];
				tmp++;
			}
			// compute the xRow
			// Ptr to first position in this line
			xRowPtr = xRows + y * imgW;

			// compute this entry
			first = next = pixelDis - winsize_div2;
			sum = 0;
			for (i = 0; i < winsize; i++)
			{
				sum += *next;
				next++;
			}
			// assign value
			*xRowPtr = sum;
			xRowPtr++;

			// the first entry was already computed, 
			// now start the incremental approach
			for (i = 1; i < imgW; i++)
			{
				sum -= *first;
				sum += *next;
				first++; next++;
				*xRowPtr = sum;
				xRowPtr++;
			}
		} // for y

		// We computed the xRows for each pixel
		// incrementally sum the up to get final sum
		for (x = 0; x < imgW; x++)
		{
			SlicePtr = SliceInD + x;
			first = next = xRows + x - winsize_div2 * imgW;
			sum = 0;
			for (i = 0; i < winsize; i++)
			{
				sum += *next;
				next += imgW;
			}
			// assign value
			*SlicePtr = sum;
			SlicePtr += imgW;

			// the first entry was already computed, 
			// now start the incremental approach
			for (y = 1; y < imgH; y++)
			{
				sum -= *first;
				sum += *next;
				first += imgW; next += imgW;
				*SlicePtr = sum;
				SlicePtr += imgW;
			}			
		} // for x

		// Copy to aggregated
		for (y = 0; y < imgH; y++)
			for (x = 0; x < imgW; x++)
				aggregated [y * imgW * (maxdisp + 1) + x * (maxdisp + 1) + d] = SliceInD[y * imgW + x] / winsize / winsize;
	}

	// free memory
	pixelDis -= winsize_div2;
	free (pixelDis);

	xRows -= winsize_div2 * imgW;
	free (xRows);

	free (SliceInD);

	return aggregated;
}