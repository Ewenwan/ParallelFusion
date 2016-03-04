#include "colourtransform.h"

float *matmul (float *mat, IplImage *input)
{
	int imgW = input->width;
	int imgH = input->height;

	float *out = (float*) malloc (3 * imgW * imgH * sizeof (float));

	float rgb[3];
	float result[3];

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			// IplImage stores BGR
			for (int i = 0; i < 3; i++)
				rgb[i] = (float) (uchar) input->imageData [y * input->widthStep + 3 * x + 2 - i];

			result[0] = mat[0] * rgb[0] + mat[1] * rgb[1] + mat[2] * rgb[2];
			result[1] = mat[3] * rgb[0] + mat[4] * rgb[1] + mat[5] * rgb[2];
			result[2] = mat[6] * rgb[0] + mat[7] * rgb[1] + mat[8] * rgb[2];

			for (int i = 0; i < 3; i++)
				out[y * 3 * imgW + 3 * x + i] = result[i];
		}

	return out;
}

float *RGB_2_GREY (IplImage *input)
{
	int imgW = input->width;
	int imgH = input->height;

	float *out = (float*) malloc (3 * imgW * imgH * sizeof (float));

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			float R = (float) (uchar) input->imageData [y * input->widthStep + 3 * x + 2];
			float G = (float) (uchar) input->imageData [y * input->widthStep + 3 * x + 1];
			float B = (float) (uchar) input->imageData [y * input->widthStep + 3 * x + 0];
			
			float grey = 0.299 * R + 0.587 * G + 0.114 * B;

			for (int i = 0; i < 3; i++)
				out[y * 3 * imgW + 3 * x + i] = grey;
		}

	return out;
}

float *RGB_2_RGB (IplImage *input)
{
	float mat[9];

	// unity matrix
	// 1 0 0
	// 0 1 0
	// 0 0 1

	for (int i = 0; i < 9; i++)
		mat[i] = 0.;

	mat[0] = mat [4] = mat[8] = 1.;

	return matmul (mat, input);
}

float *RGB_2_I1I2I3 (IplImage *input)
{
	float mat[9];
	//  1/3  1/3  1/3
	//  1/2   0  -1/2
	// -1/4 -1/4  1/2

	mat[0] = mat [1] = mat[2] = 1./3.;
	mat[3] = mat[8] = 1./2.;
	mat[4] = 0.;
	mat[5] = - 1./2.;
	mat[6] = mat[7] = - 1./4.;

	return matmul (mat, input);
}

float *RGB_2_XYZ (IplImage *input)
{
	float mat[9];
	// 0.607 0.174 0.200;
	// 0.299 0.587 0.114;
	// 0.000 0.066 1.116;

	mat[0] = 0.607; mat[1] = 0.174; mat[2] = 0.200;
	mat[3] = 0.299; mat[4] = 0.587; mat[5] = 0.114;
	mat[6] = 0.000; mat[7] = 0.066; mat[8] = 1.116;

	return matmul (mat, input);
}

float *RGB_2_AC1C2 (IplImage *input)
{
	float mat[9];
	//  1/3          1/3         1/3
	//  sqrt(3)/2 -sqrt(3)/2      0
	// -1/2          -1/2         1

	mat[0] = 1./3.; mat[1] = 1./3.; mat[2] = 1./3.;
	mat[3] = sqrt(3.)/2.; mat[4] = -1. * sqrt(3.)/2.; mat[5] = 0.;
	mat[6] = -1./2.; mat[7] = -1./2.; mat[8] = 1.;

	return matmul (mat, input);
}

float *RGB_2_YC1C2 (IplImage *input)
{
	float mat[9];
	//  1/3          1/3         1/3
	//   1          -1/2        -1/2
	//   0        -sqrt(3)/2   sqrt(3)/2

	mat[0] = 1./3.; mat[1] = 1./3.; mat[2] = 1./3.;
	mat[3] = 1.; mat[4] = -1./2.; mat[5] = -1./2.;
	mat[6] = 0.; mat[7] = -1. * sqrt(3.)/2.; mat[8] = sqrt(3.)/2.;

	return matmul (mat, input);
}

float *RGB_2_H1H2H3 (IplImage *input)
{
	float mat[9];
	//  1            1         0
	//  1           -1         0
	// -1/2          0        -1/2

	mat[0] = 1.; mat[1] = 1.; mat[2] = 0.;
	mat[3] = 1.; mat[4] = -1.; mat[5] = 0.;
	mat[6] = -1./2.; mat[7] = 0.; mat[8] = -1./2.;

	return matmul (mat, input);
}

#define XW (94.81/100.0)
#define YW (100.0/100.0)
#define ZW (107.43/100.0)

float *RGB_2_LUV (IplImage *input)
{
	float *RGB_image = RGB_2_RGB (input);

	int imgW = input->width;
	int imgH = input->height;

	float *LUV_image = (float*) malloc (3 * imgW * imgH * sizeof (float));

	float R, G, B;
	float X, Y, Z;
	float L, u, v;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			R = RGB_image [y * 3 * imgW + 3 * x] / 255.;
			G = RGB_image [y * 3 * imgW + 3 * x + 1] / 255;
			B = RGB_image [y * 3 * imgW + 3 * x + 2] / 255;

			X = 0.607 * R + 0.174 * G + 0.2 * B;
			Y = 0.299 * R + 0.587 * G + 0.114 * B;
			Z = 0.066 * G + 1.116 * B;

			if (Y / YW > 0.008856)
				L = 116.0 * pow((Y / YW), 1./3.) - 16.;
			else
				L = 903.3 * Y / YW;

			float val = X + 15. * Y + 3. * Z;
			float valw = XW + 15. * YW + 3. * ZW;

			if (val != 0.)
			{
				u = 13. * L * ((4. * X) / val - (4. * XW) / valw);
				v = 13. * L * ((9. * Y) / val - (9. * YW) / valw);
			}
			else
			{
				u = 13. * L * -1. * (4. * XW) / valw;
				v = 13. * L * -1. * (9. * YW) / valw;
			}

			//printf ("(%.3f,%.3f,%.3f)\n", L, u, v);

			LUV_image [y * 3 * imgW + 3 * x + 0] = L;
			LUV_image [y * 3 * imgW + 3 * x + 1] = u;
			LUV_image [y * 3 * imgW + 3 * x + 2] = v;
		}

	free (RGB_image);
	return LUV_image;
}

float LAB_f (float x)
{
	if (x > 0.008856)
	{
		return pow (x, (float) 1. / (float) 3.); 
	}
	else
		return 7.787 * x + (16. / 116);
}

float *RGB_2_LAB (IplImage *input)
{
	float *RGB_image = RGB_2_RGB (input);

	int imgW = input->width;
	int imgH = input->height;

	float *LAB_image = (float*) malloc (3 * imgW * imgH * sizeof (float));

	float R, G, B;
	float X, Y, Z;
	float L, a, b;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			R = RGB_image [y * 3 * imgW + 3 * x] / 255.;
			G = RGB_image [y * 3 * imgW + 3 * x + 1] / 255;
			B = RGB_image [y * 3 * imgW + 3 * x + 2] / 255;

			X = 0.607 * R + 0.174 * G + 0.2 * B;
			Y = 0.299 * R + 0.587 * G + 0.114 * B;
			Z = 0.066 * G + 1.116 * B;

			if (Y / YW > 0.008856)
				L = 116.0 * pow((Y / YW), 1. / 3.) - 16.;
			else
				L = 903.3 * Y / YW;

			a = 500. * LAB_f(X / XW) - LAB_f(Y / YW);
			b = 200. * LAB_f(Y / YW) - LAB_f(Z / ZW);

			//printf ("[%f %f %f]\n", L, a, b);

			LAB_image [y * 3 * imgW + 3 * x + 0] = L;
			LAB_image [y * 3 * imgW + 3 * x + 1] = a;
			LAB_image [y * 3 * imgW + 3 * x + 2] = b;
		}

	free (RGB_image);
	return LAB_image;
}

/*float min (float v1, float v2)
{
	if (v1 < v2)
		return v1;
	else
		return v2;
}*/

float min3(float R, float G, float B )
{
	float temp1, temp2;
	
	temp1 = min(R , G);
	temp2 = min(temp1 , B);

	return temp2;	
}

float *RGB_2_HSI (IplImage *input)
{
	int imgW = input->width;
	int imgH = input->height;

	float *HSI_image = (float*) malloc (3 * imgW * imgH * sizeof (float));

	const float PI = 3.14159265;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			float R = (float) (uchar) input->imageData [y * input->widthStep + 3 * x + 2];
			float G = (float) (uchar) input->imageData [y * input->widthStep + 3 * x + 1];
			float B = (float) (uchar) input->imageData [y * input->widthStep + 3 * x + 0];

			float I = ( R + G + B ) / 3.;

			float S;

			float denominator = (R + G + B);
			if (denominator)
				S = 1. - 3. * (min3(R, G, B) / denominator);
			else
				S = 1.;

			denominator = (R - G) * (R - G) + (R - B) * (G - B);

			float H1;

			if (denominator) 
				H1 = ((R - G) + (R - B)) / ( 2.0 * sqrt(denominator));
			else
				H1 = 0.;

			float H;
			if (B <= G)
				H = acos(H1);
			else
				H = 2. * PI - acos(H1);

			HSI_image[y * 3 * imgW + 3 * x + 0] = H;
			HSI_image[y * 3 * imgW + 3 * x + 1] = S;
			HSI_image[y * 3 * imgW + 3 * x + 2] = I;
		}

	return HSI_image;
}

float *RGB_2_LUV_SCALED (IplImage *input, float scale_LUV)
{
	int imgW = input->width;
	int imgH = input->height;

	float *LUV_image = RGB_2_LUV (input);

	for (int i = 0; i < 3 * imgW * imgH; i++)
		LUV_image[i] *= scale_LUV;

	return LUV_image;
}

float *RGB_2_LAB_SCALED (IplImage *input, float scale_LAB)
{
	int imgW = input->width;
	int imgH = input->height;

	float *LAB_image = RGB_2_LAB (input);

	for (int i = 0; i < 3 * imgW * imgH; i++)
		LAB_image[i] *= scale_LAB;

	return LAB_image;
}

float *RGB_2_CMY (IplImage *input)
{
	int imgW = input->width;
	int imgH = input->height;

	float *CMY_image = (float*) malloc (3 * imgW * imgH * sizeof (float));

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			float R = (float) (uchar) input->imageData [y * input->widthStep + 3 * x + 2];
			float G = (float) (uchar) input->imageData [y * input->widthStep + 3 * x + 1];
			float B = (float) (uchar) input->imageData [y * input->widthStep + 3 * x + 0];

			CMY_image[y * 3 * imgW + 3 * x + 0] = 255. - R;
			CMY_image[y * 3 * imgW + 3 * x + 1] = 255. - G;
			CMY_image[y * 3 * imgW + 3 * x + 2] = 255. - B;
		}

	return CMY_image;
}

float *colour_transform (IplImage *input, int color_model)
{
	float scale_LUV = 0.0;
	float scale_LAB = 2.0;

	if (color_model == LUV_SCALED)
		scale_LUV = 3.0;

	if (color_model == LUV_SCALED5)
	{
		scale_LUV = 5.0;
		color_model = LUV_SCALED;
	}

	if (color_model == LUV_SCALED10)
	{
		scale_LUV = 10.0;
		color_model = LUV_SCALED;
	}


	switch (color_model)
	{
		case GREY:
			printf ("GREY Transform\n");
			return RGB_2_GREY (input);
		break;
		case RGB:
			printf ("RGB Transform\n");
			return RGB_2_RGB (input);
		break;
		case I1I2I3:
			printf ("I1I2I3 Transform\n");
			return RGB_2_I1I2I3 (input);
		break;
		case XYZ:
			printf ("XYZ Transform\n");
			return RGB_2_XYZ (input);
		break;
		case AC1C2:
			printf ("AC1C2 Transform\n");
			return RGB_2_AC1C2 (input);
		break;
		case YC1C2:
			printf ("YC1C2 Transform\n");
			return RGB_2_YC1C2 (input);
		break;
		case H1H2H3:
			printf ("H1H2H3 Transform\n");
			return RGB_2_H1H2H3 (input);
		break;
		case LUV:
			printf ("LUV Transform\n");
			return RGB_2_LUV (input);
		break;
		case LAB:
			printf ("LAB Transform\n");
			return RGB_2_LAB (input);
		break;
		case HSI:
			printf ("HSI Transform\n");
			return RGB_2_HSI (input);
		break;
		case LUV_SCALED:
			printf ("LUV * %.1f Transform\n", scale_LUV);
			return RGB_2_LUV_SCALED (input, scale_LUV);
		break;
		case LAB_SCALED:
			printf ("LAB * %.1f Transform\n", scale_LAB);
			return RGB_2_LAB_SCALED (input, scale_LAB);
		break;
		case CMY:
			printf ("CMY Transform\n");
			return RGB_2_CMY (input);
		break;

		default:
			exit(-1);
	}
}