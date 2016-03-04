#include "pixeldis.h"

#define MIN2(x,y) ((x) < (y)? (x): (y)) 

float *CDsi::scale_up (float *scanline)
{
	float *upscaled = (float *) malloc (3 * (2 * (imgW + 1)) * sizeof (float));

	//copy pixels in every second cell
	float *sl_data = scanline;
	float *us_data = upscaled + 3;

	for (int x = 0; x < imgW; x++)
	{
		*us_data++ = *sl_data++;
		*us_data++ = *sl_data++;
		*us_data++ = *sl_data++;
		us_data += 3;
	}

	// first pixel is left interpolation pixel from border
	// set to colour of first pixel
	float *cur = upscaled;
	float *right = upscaled + 3;
	*cur++ = *right++;
	*cur++ = *right++;
	*cur++ = *right++;

	// interpolate from left and right
	float *left = upscaled + 3;
	cur = upscaled + 6;
	right = upscaled + 9;

	for (int x = 0; x < imgW - 1; x++)
	{
		*cur++ = (*left++ + *right++) / 2;
		*cur++ = (*left++ + *right++) / 2;
		*cur++ = (*left++ + *right++) / 2;
		left += 3;
		cur += 3;
		right +=3;
	}

	// right border pixel
	// copy colour from leftmost pixel
	*cur++ = *left++;
	*cur++ = *left++;
	*cur++ = *left++;

	return upscaled;
}

void CDsi::match_scanlines (float *left_data, float *right_data, int *dsi_data)
{
	int x, d, border;
	float minAD, tmp;
	float *ref, *match;

	float *left_upscaled = scale_up (left_data);
	float *right_upscaled = scale_up (right_data);
	int *dsi = dsi_data;

	if (!leftisReference)
	{
		float *dummy = left_upscaled;
		left_upscaled = right_upscaled;
		right_upscaled = dummy;
	}

	for (x = 0; x < imgW; x++)
	{
		ref = left_upscaled + 6 * x + 3;
		match = right_upscaled + 6 * x + 3;

		if (leftisReference) border = MIN2 ((maxdisp + 1), (x + 1));
		else border = MIN2 ((maxdisp + 1), (imgW - x));

		for (d = 0; d < border; d++)
		{
			if (useBT == 0)
			{
				minAD = differencemeasure (ref, match, difmeasure_method);
			}
			else
			{

				minAD = differencemeasure (ref, match, difmeasure_method); //AD(ref, match);
				tmp = differencemeasure (ref, match - 3, difmeasure_method); //AD(ref, (match - 3));
				minAD = MIN2(minAD, tmp);
				tmp = differencemeasure (ref, match + 3, difmeasure_method); //AD(ref, (match + 3));
				minAD = MIN2(minAD, tmp);
				
				// symmetry
				tmp = differencemeasure (ref - 3, match, difmeasure_method); //AD((ref - 3), match);
				minAD = MIN2(minAD, tmp);
				tmp = differencemeasure (ref + 3, match, difmeasure_method); //AD((ref + 3), match);
				minAD = MIN2(minAD, tmp);
			}

			// regulate the influence of occluded pixels
			if (minAD > pixdisthresh)
				minAD = pixdisthresh;

			*dsi++ = (int) (minAD + 0.5);
			
			if (leftisReference) match -= 6;
			else match += 6;
		}

		// image borders
		for (; d < maxdisp + 1; d++)
			*dsi++ = border_costs;
	}
	
	free (left_upscaled);
	free (right_upscaled);
}

void CDsi::change_view (int start_sl, int end_sl)
{
	int x, y, d;
	int x_step = maxdisp + 1;

	int *dsi_it = dsi_left + start_sl * imgW * x_step;
	int *dsi_right_it;
	int x_right;

	for (y = start_sl; y <= end_sl; y++)
	{
		dsi_right_it = dsi_right + y * imgW * x_step;

		for (x = 0; x < imgW; x++)
			for (d = 0; d < maxdisp + 1; d++)
			{
				x_right = x - d;
				if (x_right >= 0)
				{
					*dsi_it = *(dsi_right_it + x_right * x_step + d);
				}
				else 
					*dsi_it = border_costs;
				
				dsi_it++;
			}
	}
}

CDsi::CDsi (int _imgW, int _imgH, int _maxdisp, int _numthreads)
{
	imgW = _imgW;
	imgH = _imgH;
	maxdisp = _maxdisp;
	numthreads = _numthreads;
}

void CDsi::Generate_DSI (IplImage *_left, IplImage *_right, bool _leftisReference, int _colour_space, int _radiotransf, int _difmeasure_method, int _border_costs, int _pixdisthresh, int _useBT, int *_dsi)
{
	left = _left;
	right = _right;
	leftisReference = _leftisReference;

	colour_space = _colour_space;
	radiotransf = _radiotransf;
	difmeasure_method = _difmeasure_method;
	if (colour_space == HSI) 
		difmeasure_method = HSI_DIF;

	border_costs = _border_costs;
	pixdisthresh = _pixdisthresh;
	useBT = _useBT;

	dsi = _dsi;

	float *col_conv_left = colour_transform (left, colour_space);
	float *col_conv_right = colour_transform (right, colour_space);

	float *transf_left = radiotransform (col_conv_left, imgW, imgH, radiotransf);
	float *transf_right = radiotransform (col_conv_right, imgW, imgH, radiotransf);

	for (int y = 0; y < imgH; y++)
		match_scanlines (transf_left + y * 3 * imgW, transf_right + y * 3 * imgW, dsi + y * imgW * (maxdisp + 1));

	int aggregate_costs = 0;
	if (aggregate_costs)
	{
		int *aggregated = aggregate (dsi, imgW, imgH, maxdisp, 15);
		memcpy (dsi, aggregated, imgW * imgH * (maxdisp + 1) * sizeof (int));
	}
}

void CDsi::DSI_Left_View (int *_dsi_right, int *_dsi_left)
{
	dsi_right = _dsi_right;
	dsi_left = _dsi_left;

	#define PARALLEL_VIEW_CHANGE
	#ifdef PARALLEL_VIEW_CHANGE
//		omp_set_num_threads(numthreads);
//		#pragma omp parallel for
		for (int i = 0; i < numthreads; i++)
		{
			// partition the scanlines
			int num_sl = imgH / numthreads;
			int y_start = i * num_sl;
			int y_end = y_start + num_sl - 1;
			if (i == numthreads - 1)
				y_end = imgH - 1;			

			// run parallel dsi computation
			change_view (y_start, y_end);
			//FillDSI_right_reference (col_conv_right, col_conv_left, HMI_Scores_Lookup, y_start, y_end, dsi);
		}
//		#pragma omp barrier
	#else
		change_view (0, imgH - 1);
	#endif
}





////////////////////////////////////////////////////////////////////////////////////

void CHmiDsi::Generate_DSI (IplImage *_left, IplImage *_right, IplImage *left0, IplImage *right0, IplImage *disp0, IplImage *occmask0, bool _leftisReference, int _colour_space, int _bordercosts, int *_dsi)
{
	printf ("computing data term\n");

	left = _left;
	right = _right;
	leftisReference = _leftisReference;
	dsi = _dsi;
	border_costs = _bordercosts;

	colour_space = _colour_space;

	// colour transform
	float *scaled_col_conv_left, *scaled_col_conv_right, *col_conv_left, *col_conv_right;

	/*omp_set_num_threads(numthreads);
	#pragma omp sections
	{
		#pragma omp section
		scaled_col_conv_left = colour_transform (left0, colour_space);
		#pragma omp section
		scaled_col_conv_right = colour_transform (right0, colour_space);
		#pragma omp section
		col_conv_left = colour_transform (left, colour_space);
		#pragma omp section
		col_conv_right = colour_transform (right, colour_space);
	}*/

	scaled_col_conv_left = colour_transform (left0, colour_space);
	scaled_col_conv_right = colour_transform (right0, colour_space);
	col_conv_left = colour_transform (left, colour_space);
	col_conv_right = colour_transform (right, colour_space);

	// warp left image into geometry of right view
	float *warped0 = (float*) malloc (left0->width * left0->height * 3 * sizeof (float));
	IplImage *occmask0_right_view = cvCloneImage (occmask0);
	
	WarpToRightView (scaled_col_conv_left, disp0, occmask0, warped0, occmask0_right_view);
	
	IplImage *HMI_Scores_Lookup[3];
	ComputeMiScores (scaled_col_conv_right, warped0, occmask0_right_view, HMI_Scores_Lookup);

	free (scaled_col_conv_left);
	free (scaled_col_conv_right);
	free (warped0);
	cvReleaseImage (&occmask0_right_view);

	if (leftisReference == true)
	{
		for (int y = 0; y < imgH; y++)
			FillDSI_left_reference (col_conv_left, col_conv_right, HMI_Scores_Lookup, y, dsi);
	}
	else
	{
		//printf ("DSI Loop started\n");
//		omp_set_num_threads(numthreads);
//		#pragma omp parallel for
		for (int i = 0; i < numthreads; i++)
		{
			// partition the scanlines
			int num_sl = imgH / numthreads;
			int y_start = i * num_sl;
			int y_end = y_start + num_sl - 1;
			if (i == numthreads - 1)
				y_end = imgH - 1;			

			// run parallel dsi computation
			FillDSI_right_reference (col_conv_right, col_conv_left, HMI_Scores_Lookup, y_start, y_end, dsi);
		}
//		#pragma omp barrier
		//printf ("DSI Loop finished\n");
	}

	free (col_conv_left);
	free (col_conv_right);
	for (int i = 0; i < 3; i++)
		cvReleaseImage (&(HMI_Scores_Lookup[i]));

	int aggregate_costs = 0;
	if (aggregate_costs)
	{
		int *aggregated = aggregate (dsi, imgW, imgH, maxdisp, 3);
		memcpy (dsi, aggregated, imgW * imgH * (maxdisp + 1) * sizeof (int));
	}

	printf ("data term computed\n");
}

void showfloatarray (float *data, int imgW, int imgH, int numchannels, float scale, CvPoint pos, char *txt, float addvalue = 0.0)
{
	IplImage *show;

	if (numchannels == 1)
		show = cvCreateImage (cvSize(imgW, imgH), IPL_DEPTH_8U, 1);
	else if (numchannels == 3)
		show = cvCreateImage (cvSize(imgW, imgH), IPL_DEPTH_8U, 3);

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
			if (numchannels == 1)
			{
				int val = (int) ((data[y * imgW + x] + addvalue) * scale) + 0.5;
				if (val < 0) val *= -1;
				if (val > 255) val = 255;
				show->imageData [y * show->widthStep + x] = val;
			}
			else if (numchannels == 3)
				for (int c = 0; c < 3; c++)
				{
					int val = (int) ((data[(y * imgW + x) * 3 + 2 - c] + addvalue) * scale) + 0.5;
					if (val < 0) val *= -1;
					if (val > 255) val = 255;
					show->imageData [y * show->widthStep + 3 * x + c] = val;
				}

	cvvNamedWindow (txt, CV_WINDOW_AUTOSIZE);
	cvMoveWindow (txt, pos.x, pos.y);
	cvShowImage (txt, show);
	
	cvReleaseImage (&show);
}

void CHmiDsi::WarpToRightView (float *left0, IplImage *disp0, IplImage *occmask0, float *warped0, IplImage *occmask0_right_view)
{
	int scaled_imgW = disp0->width;
	int scaled_imgH = disp0->height;

	IplImage *disp0_right = cvCloneImage (disp0);
	cvZero (disp0_right);
	cvZero (occmask0_right_view);

	// warp to derive disparity map of second view
	for (int y = 0; y < scaled_imgH; y++)
		for (int x = 0; x < scaled_imgW; x++)
		{
			int d = (int) (uchar) disp0->imageData[y * disp0->widthStep + x];

			if (x - d < 0 || (occmask0->imageData [y * occmask0->widthStep + x]) == 0) continue;

			int curdisp_right = (int) (uchar) disp0_right->imageData[y * disp0_right->widthStep + x - d];
			if (curdisp_right < d)
			{
				disp0_right->imageData[y * disp0_right->widthStep + x - d] = (uchar) d;
				occmask0_right_view->imageData[y * occmask0_right_view->widthStep + x - d] = (uchar) 255;
			}
		}
	
	// use disparity map of second view to generate warped right view
	float *dst = warped0;
	float *src;

	for (int y = 0; y < scaled_imgH; y++)
		for (int x = 0; x < scaled_imgW; x++)
		{
			if (occmask0_right_view->imageData [y * occmask0_right_view->widthStep + x] != 0)
			{
				int d = (int) (uchar) disp0_right->imageData[y * disp0_right->widthStep + x];
				src = left0 + (y * disp0_right->width + x + d) * 3;
				*dst = *src; dst++; src++;
				*dst = *src; dst++; src++;
				*dst = *src; dst++;
			}
			else
			{
				*dst = 0.f; dst++;
				*dst = 0.f; dst++;
				*dst = 0.f; dst++;
			}		
		}

	#ifdef SHOW_HMI_STEPS
		cvConvertScale(disp0_right, disp0_right, 16);
		cvvNamedWindow ("disp_right", CV_WINDOW_AUTOSIZE);
		cvMoveWindow ("disp_right", disp0_right->width, 0);
		cvShowImage ("disp_right", disp0_right);
		cvvNamedWindow ("occmask0_right", CV_WINDOW_AUTOSIZE);
		cvMoveWindow ("occmask0_right", 0, 0);
		cvShowImage ("occmask0_right", occmask0_right_view);
	#endif

	cvReleaseImage (&disp0_right);
}

void convolve_gaussian (IplImage *src, IplImage *dst)
{
	cvSmooth( src, dst, CV_GAUSSIAN, 7, 7);
}

void find_largest_val (IplImage *src, float &largest_val)
{
	largest_val = 0.0;
	float *data = (float*) src->imageData;

	for (int i = 0; i < src->width * src->height; i++)
	{
		if (*data > largest_val)
			largest_val = *data;
		data++;
	}
}

void minus_log (IplImage *src, IplImage *dst, float largest_val)
{
	int width = src->width;
	int height = src->height;
	float *src_data = (float*) src->imageData;
	float *dst_data = (float*) dst->imageData;

	float log_lv;

	if (largest_val == 0.0)
		log_lv = log (0.0001);
	else
		log_lv = log (largest_val);

	for (int i = 0; i < width * height; i++)
	{
		if (*src_data == 0.)
			*src_data = 0.0001;

		*dst_data = -log(*src_data) + log_lv; 

		dst_data++;
		src_data++;
	}

}



void CHmiDsi::ComputeMiScores (float *right0, float *warped0, IplImage *occmask0_right_view, IplImage **HMI_Scores_Lookup)
{
	int width = occmask0_right_view->width;
	int height = occmask0_right_view->height;

	int coldim_x = 256;
	int coldim_y = 256;

	//omp_set_num_threads(numthreads);
	//#pragma omp parallel for
	for (int c = 0; c < 3; c++)
	{
		HMI_Scores_Lookup[c] = cvCreateImage(cvSize(coldim_x, coldim_y), IPL_DEPTH_32F, 1);
		cvZero (HMI_Scores_Lookup[c]);

		float *data = (float*) HMI_Scores_Lookup[c]->imageData;

		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
			{
				if (occmask0_right_view->imageData[y * occmask0_right_view->widthStep + x] == 0)
					continue;

				int val_right = (int) (right0[(y * width + x) * 3 + c] + 0.5);
				int val_warped = (int) (warped0[(y * width + x) * 3 + c] + 0.5);

				data[val_right * coldim_x + val_warped]++;
			}

		#ifdef SHOW_HMI_STEPS
			showfloatarray (right0, width, height, 3, 1.0, cvPoint(0,height), "right0");
			showfloatarray (warped0, width, height, 3, 1.0, cvPoint(width,height), "warped0");
			showfloatarray ((float*)HMI_Scores_Lookup[c]->imageData, coldim_x, coldim_y, 1, 50.0, cvPoint(0,2*height), "Histogram");
		#endif

		// convolution
		convolve_gaussian (HMI_Scores_Lookup[c], HMI_Scores_Lookup[c]);
		#ifdef SHOW_HMI_STEPS
			showfloatarray ((float*)HMI_Scores_Lookup[c]->imageData, coldim_x, coldim_y, 1, 50.0, cvPoint(coldim_x,2*height), "g conv P");
		#endif

		// find the largest value in the array
		float largest_val;
		find_largest_val (HMI_Scores_Lookup[c], largest_val);

		// negative log
		minus_log (HMI_Scores_Lookup[c], HMI_Scores_Lookup[c], largest_val);
		#ifdef SHOW_HMI_STEPS
			showfloatarray ((float*)HMI_Scores_Lookup[c]->imageData, coldim_x, coldim_y, 1, 1.0, cvPoint(2*coldim_x,2*height), "minus log");
		#endif
		
		// convolution
		convolve_gaussian (HMI_Scores_Lookup[c], HMI_Scores_Lookup[c]);

		#ifdef SHOW_HMI_STEPS
			showfloatarray ((float*)HMI_Scores_Lookup[c]->imageData, coldim_x, coldim_y, 1, 1.0, cvPoint(3*coldim_x,2*height), "conv P minus log");
			cvWaitKey(0);
		#endif
	}
	//#pragma omp barrier
}

void CHmiDsi::FillDSI_left_reference (float *left, float *right, IplImage **HMI_Scores_Lookup, int y, int *dsi)
{
	//printf ("DSI multiplied by 10\n");

	int *dsi_data = dsi + y * imgW * (maxdisp + 1);


	float *lookup[3];
	lookup[0] = (float*) HMI_Scores_Lookup[0]->imageData;
	lookup[1] = (float*) HMI_Scores_Lookup[1]->imageData;
	lookup[2] = (float*) HMI_Scores_Lookup[2]->imageData;

	int coldim = 256;

	float *data_left = left + 3 * y * imgW;
	float *scanline_start_right = right + 3 * y * imgW;

	int I_left[3], I_right[3];
	float *matchpoint_right;

	for (int x = 0; x < imgW; x++)
	{
		for (int c = 0; c < 3; c++)
		{
			I_left[c] = (int) *data_left; 
			data_left++;
		}

		//int min_costs = 99999;

		for (int d = 0; d < maxdisp + 1; d++)
		{
			if (x - d >= 0)
			{
				float sum = 0.0;
				
				matchpoint_right = scanline_start_right + 3 * (x - d);

				for (int c = 0; c < 3; c++)
				{
					I_right[c] = (int) *matchpoint_right;

					sum += lookup[c][I_right[c] * coldim + I_left[c]];

					matchpoint_right++;
				}

				/////////////////////
				// time 10 truncated by 200
				*dsi_data = (int) (sum * 1000.0);

				//if (x == 189 && y == 36)
				//	printf ("d %d %.3f => %d\n", d, sum, *dsi_data);
			

				/*int trunc = 125;
				if (*dsi_data > trunc)
					*dsi_data = trunc;*/
				/////////////////////
				// quadratic
				//*dsi_data = (int) (sum * sum);


				dsi_data++;
			}
			else
			{
				*dsi_data = border_costs;
				dsi_data++;
			}

			//if (*(dsi_data - 1) < min_costs)
			//	min_costs = *(dsi_data - 1);
		}

		//printf ("(%d,%d) %d\n", x, y, min_costs);
	}
}

void CHmiDsi::FillDSI_right_reference (float *right, float *left, IplImage **HMI_Scores_Lookup, int y_start, int y_end, int *dsi)
{
	int *dsi_data = dsi + y_start * imgW * (maxdisp + 1);

	float *data_right, *scanline_start_left, *matchpoint_left;
	int I_right_times_coldim[3], I_left[3];

	float *lookup[3];
	lookup[0] = (float*) HMI_Scores_Lookup[0]->imageData;
	lookup[1] = (float*) HMI_Scores_Lookup[1]->imageData;
	lookup[2] = (float*) HMI_Scores_Lookup[2]->imageData;

	int coldim = 256;

	for (int y = y_start; y <= y_end; y++)
	{
		//printf ("scanline %d processed by thread %d\n", y, omp_get_thread_num());

		data_right = right + 3 * y * imgW;
		scanline_start_left = left + 3 * y * imgW;

		for (int x = 0; x < imgW; x++)
		{
			I_right_times_coldim[0] = ((int) *data_right) * coldim; data_right++;
			I_right_times_coldim[1] = ((int) *data_right) * coldim; data_right++;
			I_right_times_coldim[2] = ((int) *data_right) * coldim; data_right++;

			for (int d = 0; d < maxdisp + 1; d++)
			{
				if (x + d < imgW)
				{
					float sum = 0.0;
					
					matchpoint_left = scanline_start_left + 3 * (x + d);
					I_left[0] = (int) *matchpoint_left; matchpoint_left++;
					I_left[1] = (int) *matchpoint_left; matchpoint_left++;
					I_left[2] = (int) *matchpoint_left;
					
					sum = lookup[0][I_right_times_coldim[0] + I_left[0]] + 
						  lookup[1][I_right_times_coldim[1] + I_left[1]] +
						  lookup[2][I_right_times_coldim[2] + I_left[2]];

					*dsi_data = (int) (sum * 10.0);

					dsi_data++;
				}
				else
				{
					*dsi_data = border_costs;
					dsi_data++;
				}
			}
		}
	}
}

inline uchar *get_padded_dsi_ptr_at (uchar *padded_dsi, int x, int y, int d, int padded_imgW, int padded_imgH, int maxdisp, int paddingpix)
{
	return padded_dsi + d * padded_imgW * padded_imgH + (y + paddingpix) * padded_imgW + x + paddingpix;
}

void Gererate_right_disp_and_occmask (IplImage *disp, IplImage *&disp_transformed, IplImage *&occmask)
{
	int imgW = disp->width;
	int imgH = disp->height;

	occmask = cvCloneImage (disp);
	disp_transformed = cvCloneImage (disp);
	cvZero (occmask);
	cvZero (disp_transformed);

	// warp to derive disparity map of second view
	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			int d = (int) (uchar) disp->imageData[y * disp->widthStep + x];

			if (x - d < 0) 
				continue;

			int curdisp_right = (int) (uchar) disp_transformed->imageData[y * disp_transformed->widthStep + x - d];
			if (curdisp_right < d)
			{
				disp_transformed->imageData[y * disp_transformed->widthStep + x - d] = (uchar) d;
				occmask->imageData[y * occmask->widthStep + x - d] = (uchar) 255;
			}
		}
}

IplImage *GenerateOccMask (IplImage *disp)
{
	int imgW = disp->width;
	int imgH = disp->height;

	IplImage *occmask = cvCloneImage (disp);
	IplImage *warped = cvCloneImage (disp);
	cvZero (occmask);
	cvZero (warped);

	// warp to derive disparity map of second view
	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			int d = (int) (uchar) disp->imageData[y * disp->widthStep + x];

			if (x - d < 0) 
				continue;

			int curdisp_right = (int) (uchar) warped->imageData[y * warped->widthStep + x - d];
			if (curdisp_right < d)
			{
				warped->imageData[y * warped->widthStep + x - d] = (uchar) d;
			}
		}

	// now check whether disparity values of disp and warped disp agree
	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
		{
			int d = (int) (uchar) disp->imageData[y * disp->widthStep + x];

			if (x - d < 0) 
			{
				occmask->imageData[y * occmask->widthStep + x] = (uchar) 0;
				continue;
			}

			int curdisp_right = (int) (uchar) warped->imageData[y * warped->widthStep + x - d];

			if (d != curdisp_right)
				occmask->imageData[y * occmask->widthStep + x] = (uchar) 0;
			else
				occmask->imageData[y * occmask->widthStep + x] = (uchar) 255;

		}

/*	cvvNamedWindow ("warped", CV_WINDOW_AUTOSIZE);
	cvMoveWindow ("warped", 0, 0);
	cvShowImage ("warped", warped);

	cvvNamedWindow ("occmask", CV_WINDOW_AUTOSIZE);
	cvMoveWindow ("occmask", imgW, 0);
	cvShowImage ("occmask", occmask);

	cvWaitKey(0);*/

	return occmask;
}

/*int min (int v1, int v2)
{
	if (v1 < v2)
		return v1;
	else
		return v2;
}*/

void comp_dsi (IplImage *left_img, IplImage *right_img, int maxdisp, int winSize, uchar *&dsi, int &dsi_width, int &dsi_height, int bordercosts, int pixdisthresh, int useBT, int scale)
{
	int imgW = left_img->width;
	int imgH = left_img->height;	

	int paddingpix = (int) winSize / 2;

	// The dsi is padded to handle border pixels
	dsi_width = imgW + 2 * paddingpix;
	dsi_height = imgH + 2 * paddingpix;

	// The dsi width muss be dividable by 16 to allow for fast float4 reads on the GPU
	// CUDA is most efficient for 128 Bit loads
	int modulo_val = dsi_width % 16;
	if (modulo_val != 0)
		dsi_width += 16 - modulo_val;

	// compute dsi
	int *dsi_int = (int*) _mm_malloc (imgW * imgH * (maxdisp + 1) * sizeof(int), 16);

	#define MI
	#ifndef MI
		CDsi dsiobj (left_img->width, left_img->height, maxdisp, 4);
		dsiobj.Generate_DSI (left_img, right_img, TRUE, RGB_, RANK_TRANSFORM_35, L1, bordercosts, pixdisthresh, useBT, dsi_int);
	#else
		CHmiDsi dsiobj (imgW, imgH, maxdisp, 4);
	
		IplImage *loaddisp = 0;

		if (maxdisp == 16)
		{
			printf ("\nloading Tsukuba disp\n");
			loaddisp = cvLoadImage ("MI_disps\\tsukuba_prevdisp_it1.png");
		}
		else if (maxdisp == 31)
		{
			printf ("\nloading Venus disp\n");
			loaddisp = cvLoadImage ("MI_disps\\venus_prevdisp_it1.png");
		}
		else if (maxdisp == 59)
		{
			printf ("\nloading Cones disp\n");
			loaddisp = cvLoadImage ("MI_disps\\cones_prevdisp_it1.png");
		}
		else if (maxdisp == 60)
		{
			printf ("\nloading Teddy disp\n");
			loaddisp = cvLoadImage ("MI_disps\\teddy_prevdisp_it1.png");
		}
		else if (maxdisp == 89)
		{
			printf ("\nloading Art disp\n");
			loaddisp = cvLoadImage ("MI_disps\\moebius_prevdisp_it1.png");
		}

		if (!loaddisp)
			printf ("Could not load image for Mutual Information Score\n");

		IplImage *disp, *occmask;

		disp = cvCreateImage (cvSize(imgW, imgH), IPL_DEPTH_8U, 1);

		// one channel image, divide by scale
		for (int y = 0; y < imgH; y++)
			for (int x = 0; x < imgW; x++)
			{
				float origd = (float) (uchar) loaddisp->imageData[y * loaddisp->widthStep + 3 * x];
				disp->imageData[y * disp->widthStep + x] = (char) (int) (origd / (float) scale);
			}

		// check if left or right reference
		static int left_reference = 0;
		if (!left_reference)
		{
			IplImage *disp_transformed;
			Gererate_right_disp_and_occmask (disp, disp_transformed, occmask);
			cvReleaseImage (&disp);
			disp = disp_transformed;

			cvFlip (disp_transformed, disp_transformed, 1);
			cvFlip (occmask, occmask, 1);
		}
		else
			occmask = GenerateOccMask (disp);
		left_reference++;


/*		cvvNamedWindow ("disp_transformed", CV_WINDOW_AUTOSIZE);
		cvMoveWindow ("disp_transformed", 0, 0);
		cvShowImage ("disp_transformed", disp);

		cvvNamedWindow ("occmask", CV_WINDOW_AUTOSIZE);
		cvMoveWindow ("occmask", imgW, 0);
		cvShowImage ("occmask", occmask);

		cvvNamedWindow ("left_img", CV_WINDOW_AUTOSIZE);
		cvMoveWindow ("left_img", imgW, 0);
		cvShowImage ("left_img", left_img);

		cvvNamedWindow ("right_img", CV_WINDOW_AUTOSIZE);
		cvMoveWindow ("right_img", imgW, 0);
		cvShowImage ("right_img", right_img);

		cvWaitKey(0);*/

		dsiobj.Generate_DSI (left_img, right_img, left_img, right_img, disp, occmask, true, RGB, bordercosts, dsi_int);
	#endif

	
	int padded_dsi_size = dsi_width * dsi_height * (maxdisp + 1) * sizeof(uchar);

	uchar *dsi_uchar = (uchar*) malloc (padded_dsi_size);
	// set zero
	memset (dsi_uchar, 0, padded_dsi_size);


	// we need to reorder dsi and convert it to uchar
	int* read = dsi_int;
	uchar *write = 0;
	int curval;

	for (int y = 0; y < imgH; y++)
	{
		for (int x = 0; x < imgW; x++)
		{
			for (int d = 0; d <= maxdisp; d++)
			{
				uchar *write = get_padded_dsi_ptr_at (dsi_uchar, x, y, d, dsi_width, dsi_height, maxdisp, paddingpix);		
				*write = (uchar) std::min (*read, std::min (pixdisthresh, 255));
				//*write = (uchar) min (x, 255);
				read++;
			}
		}
	}

	dsi = dsi_uchar;
}