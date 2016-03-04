#include "radiotransform.h"

float *ranktransform (float *input, int imgW, int imgH, int winsize)
{
	float *ranks = (float*) malloc (3 * imgW * imgH * sizeof (float));

	int radius = winsize / 2;
	
	int low_cnt[3];
	float center_BGR[3];

	for (int center_y = 0; center_y < imgH; center_y++)
		for (int center_x = 0; center_x < imgW; center_x++)
		{
			low_cnt[0] = low_cnt[1] = low_cnt[2] = 0;
			// get colour of center pixel
			for (int i = 0; i < 3; i++)
				center_BGR[i] = input[center_y * 3 * imgW + 3 * center_x + i];

			// compute win borders
			int down = center_y - radius - 1;
			int up = center_y + radius;
			int left = center_x - radius - 1;
			int right = center_x + radius;

			if (down < 0) down = 0;
			if (up >= imgH) up = imgH - 1;
			if (left < 0) left = 0;
			if (right >= imgW) right = imgW - 1;

			for (int y = down; y <= up; y++)
				for (int x = left; x <= right; x++)
				{
					for (int j = 0; j < 3; j++)
					{
						float val = input[y * 3 * imgW + 3 * x + j];
						if (val < center_BGR[j])
							low_cnt[j]++;
					}
				}

			for (int i = 0; i < 3; i++)
				ranks[center_y * 3 * imgW + 3 * center_x + i] = low_cnt[i] * 255. / ((up - down) * (right - left)) / 4;
		}

	return ranks;
}

float *radiotransform (float *input, int imgW, int imgH, int radiotransf)
{
	if (radiotransf >= RANK_TRANSFORM_9 && radiotransf <= RANK_TRANSFORM_35)
	// rank transforms with different window sizes
	{
		printf ("RANK Transform\n");

		int winsize = 15;
		if (radiotransf == RANK_TRANSFORM_9) winsize = 9;
		else if (radiotransf == RANK_TRANSFORM_15) winsize = 15;
		else if (radiotransf == RANK_TRANSFORM_25) winsize = 25;
		else if (radiotransf == RANK_TRANSFORM_35) winsize = 35;

		float *transf = ranktransform (input, imgW, imgH, winsize);
		free (input);
		return transf;
	}
	else
	// no radiometric transforms
	{
		printf ("NO RADIOMETRIC Transform\n");
		return input;
	}
}