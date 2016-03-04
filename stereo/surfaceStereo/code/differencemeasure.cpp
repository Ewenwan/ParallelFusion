#include "differencemeasure.h"

inline float L1_dif (float *ref, float *match)
{
	return abs (ref[0] - match[0]) + abs (ref[1] - match[1]) + abs (ref[2] - match[2]);
}

inline float L2_dif (float *ref, float *match)
{
	float sum = 0;
	float tmp;

	tmp = abs (ref[0] - match[0]);
	tmp *= tmp;
	sum += tmp;

	tmp = abs (ref[1] - match[1]);
	tmp *= tmp;
	sum += tmp;

	tmp = abs (ref[2] - match[2]);
	tmp *= tmp;
	sum += tmp;

	tmp = sqrt (sum);

	return tmp;
}

inline float minimum_dif (float *ref, float *match)
{
	float min = abs (ref[0] - match[0]);

	float tmp = abs (ref[1] - match[1]);

	if (tmp < min) min = tmp;

	tmp = abs (ref[2] - match[2]);

	if (tmp < min) min = tmp;

	return 3 * min;
}

inline float maximum_dif (float *ref, float *match)
{
	float max = abs (ref[0] - match[0]);

	float tmp = abs (ref[1] - match[1]);

	if (tmp > max) max = tmp;

	tmp = abs (ref[2] - match[2]);

	if (tmp > max) max = tmp;

	return 3 * max;
}

inline float median_dif (float *ref, float *match)
{
	float a = abs (ref[0] - match[0]);
	float b = abs (ref[1] - match[1]);
	float c = abs (ref[2] - match[2]);

	float ordered[3];
	// sort
   if (a < b)						// a < b here
   {							
		if (a < c)					// a < c     : a the smallest
		{
			if (b < c)				// b < c  : a < b < c
			{
				ordered[0] = a; ordered[1] = b; ordered[2] = c;
			}
			else					// c <= b : a < c <= b
			{
				ordered[0] = a; ordered[1] = c; ordered[2] = b;
			}
		}
		else						// a >= c    : c <= a < b
		{
			ordered[0] = c; ordered[1] = a; ordered[2] = b;
		}
   }
   else      
   {
	   if (b < c)					// b < c     : b the smallest
	   {
			if (a < c)				// a < c   : b <= a < c
			{
				ordered[0] = b; ordered[1] = a; ordered[2] = c;
			}
			else 					// a >= c  : b < c <= a
			{
				ordered[0] = b; ordered[1] = c; ordered[2] = a;
			}
		}
		else						// c <= b    : c <= b <= a
		{
			ordered[0] = c; ordered[1] = b; ordered[2] = a;
		}
   }

   if (ordered[0] > ordered[1] || ordered[1] > ordered[2])
	   printf ("error median\n");

	return 3 * ordered[1];
}

inline float belli_dif (float *ref, float *match)
{
	float sum1 = 0.;
	float sum2 = 0.;
	float tmp;

	tmp = abs (ref[0] - match[0]);
	sum1 += tmp;
	tmp *= tmp;
	sum2 += tmp;

	tmp = abs (ref[1] - match[1]);
	sum1 += tmp;
	tmp *= tmp;
	sum2 += tmp;

	tmp = abs (ref[2] - match[2]);
	sum1 += tmp;
	tmp *= tmp;
	sum2 += tmp;

	if (!sum1) return 0;

	tmp = sum2 / sum1;

	return tmp;
}

inline float hsi_dif (float *ref, float *match)
{
	const float PI = 3.14159265;

	float Hl = ref[0];
	float Sl = ref[1];
	float Il = ref[2];

	float Hr = match[0];
	float Sr = match[1];
	float Ir = match[2];

	float Hdif = fabs (Hl - Hr);
	
	float theta;
	if (Hdif <= PI)
		theta = Hdif;
	else
		theta = 2. * PI - Hdif;

	float sum = (Il - Ir) * (Il - Ir) +
				Sl * Sl + Sr * Sr -
				2. * Sl * Sr * cos(theta);

	float dif = sqrt (sum);

	if (dif > 255.) dif = 255.;

	return dif;
}

float differencemeasure (float *ref, float *match, int difmeasure_method)
{
	switch (difmeasure_method)
	{
		case L1:
			return L1_dif (ref, match);
		break;
		case L2:
			return L2_dif (ref, match);
		break;
		case MINIMUM:
			return minimum_dif (ref, match);
		break;
		case MAXIMUM:
			return maximum_dif (ref, match);
		break;
		case MEDIAN:
			return median_dif (ref, match);
		break;
		case BELLI:
			return belli_dif (ref, match);
		break;
		case HSI_DIF:
			return hsi_dif (ref, match);
		break;

		default:
			exit(-1);
	}
	return 0;
}