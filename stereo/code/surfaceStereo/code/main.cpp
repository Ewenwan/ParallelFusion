#include "QPBO.h"
#include "pixeldis.h"

#include "proposals.h"
#include "fusion.h"
#include "warp.h"

#include <string>
#include "colourtransform.h"
#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h> //_mm_malloc and _mm_free

// name of the stereo pair (e.g., Teddy)
char *name;
// left and right filenames
char *leftfn;
char *rightfn;
// disparity map filename
char *dispfn;

// maximum allowed disparity
int maxdisp;
// scale factor for visualization
int scale;

// parameters - see main() for their meaning
int OccPen; 
int SmoothPen; 
int HOPen;
int SparsePen;
vector<int> segimages;
float SurfComplexityPen;
int winsize;
int sigma;
int gc_iterations;


/////////////////////////////////////////////
//      Parsing from the command shell
void Usage()
{
	exit(-1);
}

void copystring (char *src, char *&dst)
{
	dst = (char*) malloc (512 * sizeof (char));

	int i;
	for (i = 0; i < strlen(src); i++)
		dst[i] = src[i];

	dst[i] = '\0';

	printf ("parser: %s %d\n", dst, strlen(dst));
}

void process_commandline (int argc, char **argv)
{
	if (argc == 1) return; 

	int pos = 1;

	vector<int> cmdline_segs;

	while (1)
	{
		if (pos + 1 >= argc) break;

		if (strcmp (argv[pos], "-d") == 0)
		{
			if (!sscanf (argv[++pos], "%d", &maxdisp)) Usage();
			pos++;
		}
		if (strcmp (argv[pos], "-s") == 0)
		{
			if (!sscanf (argv[++pos], "%d", &scale)) Usage();
			pos++;
		}
		else 
		if (strcmp (argv[pos], "-occ") == 0)
		{
			if (!sscanf (argv[++pos], "%d", &OccPen)) Usage();
			pos++;
		}
		else 
		if (strcmp (argv[pos], "-sigma") == 0)
		{
			if (!sscanf (argv[++pos], "%d", &sigma)) Usage();
			pos++;
		}
		else 
		if (strcmp (argv[pos], "-smooth") == 0)
		{
			if (!sscanf (argv[++pos], "%d", &SmoothPen)) Usage();
			pos++;
		}
		else 
		if (strcmp (argv[pos], "-seg") == 0)
		{
			int dummy;
			if (!sscanf (argv[++pos], "%d", &dummy)) Usage();
			cmdline_segs.push_back (dummy);
			pos++;
		}
		else
		if (strcmp (argv[pos], "-ho") == 0)
		{
			if (!sscanf (argv[++pos], "%d", &HOPen)) Usage();
			pos++;
		}
		else
		if (strcmp (argv[pos], "-winsize") == 0)
		{
			if (!sscanf (argv[++pos], "%d", &winsize)) Usage();
			pos++;
		}
		else
		if (strcmp (argv[pos], "-sparse") == 0)
		{
			if (!sscanf (argv[++pos], "%d", &SparsePen)) Usage();
			pos++;
		}
		else
		if (strcmp (argv[pos], "-surfcomp") == 0)
		{
			int dummy;
			if (!sscanf (argv[++pos], "%d", &dummy)) Usage();
			SurfComplexityPen = (float) dummy;
			pos++;
		}
		else
		if (strcmp (argv[pos], "-iterations") == 0)
		{
			if (!sscanf (argv[++pos], "%d", &gc_iterations)) Usage();
			pos++;
		}
		else
			break;
	}

	if (pos + 4 != argc) return;

	copystring (argv[pos++], name);
	copystring (argv[pos++], leftfn);
	copystring (argv[pos++], rightfn);
	copystring (argv[pos++], dispfn);

	if (cmdline_segs.size() > 0)
		segimages = cmdline_segs;
}

void printparms ()
{
	printf ("Surface Stereo CVPR2010\n");
	printf ("--------------------------\n");
	printf ("--	    Parameters       --\n");
	printf ("--------------------------\n");

	printf ("Input files:\n");
	printf ("name %s\n", name);
	printf ("leftfn %s\n", leftfn);
	printf ("rightfn %s\n", rightfn);
	printf ("dispfn %s\n", dispfn);
	printf ("maxdisp %d\n", maxdisp);
	printf ("scale %d\n", scale);

	printf ("\nData term:\n");
	printf ("OccPen %d\n", OccPen);
	printf ("sigma %d\n", sigma);

	printf ("\nSmoothness term:\n");
	printf ("SmoothPen %d\n", SmoothPen);

	printf ("\nSegmentation consistency term:\n");
	printf ("Segmentation maps ");
	for (int i = 0; i < segimages.size(); i++)
		printf ("%d ", segimages[i]);
	printf ("\n");
	printf ("winsize %d\n", winsize);
	printf ("HOPen %d\n", HOPen);

	printf ("\nSparsity term:\n");
	printf ("SparsePen %d\n", SparsePen);
	printf ("SurfComplexityPen %.2f\n", SurfComplexityPen);

	printf ("--------------------------\n");
}
/////////////////////////////////////////////



/////////////////////////////////////////////
// load left and right images
void load_images(char *leftfn, char *rightfn, IplImage *&leftim, IplImage *&rightim)
{
	leftim = cvLoadImage (leftfn);
	rightim = cvLoadImage (rightfn);

	// check if images exist
	if (!leftim)  { fprintf(stderr, "%s could not be found\n", leftfn); exit(1); }
	if (!rightim) { fprintf(stderr, "%s could not be found\n", rightfn); exit(1); }

	// check if image sizes agree
	if (leftim->width != rightim->width || leftim->height != rightim->height)
		{ fprintf(stderr, "images sizes does not agree. Press any key."); /*getch();*/ exit(1); }

	// check if color channels agree
	if (leftim->nChannels != leftim->nChannels)
		{ fprintf(stderr, "Color channels do not agree. Press any key."); /*getch();*/ exit(1); }
}
/////////////////////////////////////////////



/////////////////////////////////////////////
// construct the cost volume
void SwapImages (IplImage *&left, IplImage *&right)
// swaps and mirrors left and right images
// used to obtain right disparity map in MI computation
{
	cvFlip (left, left, 1);
	cvFlip (right, right, 1);

	IplImage *temp = left;
	left = right;
	right = temp;
}

int *GenerateDSI_HMI (IplImage *left, IplImage *right, IplImage *&occ_left, IplImage *&disp_right)
// computes the cost volume using Mutual information (MI)
{
	int imgW = left->width;
	int imgH = left->height;

	// MI requires a disparity map and an occlusion map according to which the matching scores are computed
	// we use the SimpleTree algorithm

	// parameters for simple tree algorithm
	int P1 = 40;
	int P2 = 80;

	// call simpletree to compute left disparity map
	IplImage *disp_left = ExternDisparityMap (name, left, right, maxdisp, scale, P1, P2);


	// we also need the right disparity map to get the occlusions for the left image
	char right_name[2048];
	sprintf (right_name, "%s_right", name);

	// swap left and right image and run SimpleTree to get the right disparity map
	SwapImages (left, right);
	disp_right = ExternDisparityMap (right_name, left, right, maxdisp, scale, P1, P2);
	SwapImages (left, right);

	// occlusion map for left image computed by warping right image into the geometry of the left one
	occ_left = Occ_Mask_Via_Disp_Map (disp_right, scale);

	// images need to be "mirrored"
	cvFlip (disp_right, disp_right, 1);
	cvFlip (occ_left, occ_left, 1);

	// uncomment to display images
	//#define SHOW_MI_IMAGES
	#ifdef SHOW_MI_IMAGES
		cvvNamedWindow ("median_disp_left", CV_WINDOW_AUTOSIZE);
		cvMoveWindow ("median_disp_left", 0, 0);
		cvShowImage ("median_disp_left", disp_left);

		cvvNamedWindow ("disp_right", CV_WINDOW_AUTOSIZE);
		cvMoveWindow ("disp_right", 0, imgH);
		cvShowImage ("disp_right", disp_right);

		cvvNamedWindow ("occ_left", CV_WINDOW_AUTOSIZE);
		cvMoveWindow ("occ_left", imgW, imgH);
		cvShowImage ("occ_left", occ_left);
		cvWaitKey(0);
	#endif

	// disparity are scaled by the factor scale
	// get raw disparities by removing the scale factor
	disp_left = Unscale_Image (disp_left, scale);
	disp_right = Unscale_Image (disp_right, scale);

	// construct the cost volume
	// Generate DSI
	int *dsi_left = (int*) _mm_malloc (imgW * imgH * (maxdisp + 1) * sizeof(int), 16);

	int border_costs = 8500;	
	int num_threads = 1;
	int colSp = RGB;					// definition of ColorSpace
	int difmeasure_method = L1;			// difference method
	int radio = NO_TRANSFORM;			// radiometric method

	// having left disparity map and left occlusion map compute MI scores
	// I follow Hirschm�llers Semi-global PAMI paper to do this
	CHmiDsi dsiobj (imgW, imgH, maxdisp, num_threads);
	dsiobj.Generate_DSI (left, right, left, right, disp_left, occ_left, 1, colSp, border_costs, dsi_left);

	// return cost volume
	return dsi_left;
}

void WoodfordDataCosts (int *dsi, int imgW, int imgH)
// implements the robust matching cost function of eq. (3)
{
	int i = 0;

	float trans0 = log (2.f);

	float sigmafl = (float) sigma;
	sigmafl /= 10.f;

	for (int y = 0; y < imgH; y++)
		for (int x = 0; x < imgW; x++)
			for (int d = 0; d < maxdisp + 1; d++)
			{
				float curcosts = (float) dsi[i] / (3.f * 1000.f);

				float trans = -log (1.f + exp(-1.f * curcosts * curcosts / (float) sigmafl));

				trans = (trans + trans0) / trans0 * (float) (1000 - 1);

				dsi[i] = (int) (trans + 0.5f);

				i++;	
			}
}
/////////////////////////////////////////////


int main(int argc, char **argv)
{
	///////////////////////////
	//    the different test sets
	#define TSUKUBA
	#ifdef TSUKUBA
		name = "tsukuba";
		leftfn = "data/tsukuba_left.ppm";
		rightfn = "data/tsukuba_right.ppm";
		dispfn = "results/disp.png";
		maxdisp = 15;
		scale = 16;
	#endif

	//#define VENUS
	#ifdef VENUS
		name = "venus";
		leftfn = "..\\data\\venus_left.ppm";
		rightfn = "..\\data\\venus_right.ppm";
		dispfn = "..\\results\\disp.png";
		maxdisp = 24;
		scale = 8;
	#endif

	//#define CONES
	#ifdef CONES
		name = "cones";
		leftfn = "..\\data\\cones_left.ppm";
		rightfn = "..\\data\\cones_right.ppm";
		dispfn = "..\\results\\disp.png";
		maxdisp = 59;
		scale = 4;
	#endif

	//#define TEDDY
	#ifdef TEDDY
		name = "teddy";
		leftfn = "..\\data\\teddy_left.ppm";
		rightfn = "..\\data\\teddy_right.ppm";
		dispfn = "..\\results\\disp.png";
		maxdisp = 59;
		scale = 4;
	#endif

	//#define MAP
	#ifdef MAP
		name = "map";
		leftfn = "..\\data\\map_left.ppm";
		rightfn = "..\\data\\map_right.ppm";
		dispfn = "..\\results\\disp.png";
		maxdisp = 32;
		scale = 8;
	#endif

	//#define CLOTH3
	#ifdef CLOTH3
		name = "cloth3";
		leftfn = "..\\data\\cloth3_left.png";
		rightfn = "..\\data\\cloth3_right.png";
		dispfn = "..\\results\\disp.png";
		maxdisp = 80;
		scale = 3;
	#endif

	// occlusion penalty (eq. (2))
	OccPen = 1000; 
	// parameter for robust dissimilarity function (eq. (3))
	sigma = 50;
	
	// pairwise smoothness costs (eq. (5))
	SmoothPen = 60; 

	// segment consistency term
	// size of the window
	winsize = 5;
	// represents $\lambda_{seg}$ in eq. (7)
	HOPen = 100;
	// the mean shift segmentation images need to be precomputed (run edison)
	// we load
	segimages.push_back (3);


	// curvature term (penalty on curvature of splines)
	// represents $\lambda_{curv}$ in eq. (8)
	SurfComplexityPen = 7.0f;

	// mdl term
	// represents $\lambda_{mdl}$ in eq. (9)
	SparsePen = 20000;

	// we run 3 iterations through the proposal sequence (see 4.2. Proposal Generation)
	gc_iterations = 3;


	///////////////////////////
	process_commandline (argc, argv);
	printparms ();
	///////////////////////////
	IplImage *left = 0;
	IplImage *right = 0;

	// load
	load_images (leftfn, rightfn, left, right);

	// compute cost volume using Mutual Information
	IplImage *occ_left, *disp_right;
	int *dsi_left = GenerateDSI_HMI (left, right, occ_left, disp_right);

	// robust function of eq. (3)
	WoodfordDataCosts (dsi_left, left->width, left->height);

	// init the proposal generator
	ProposalGenerator generator (name, left, right, maxdisp, scale, gc_iterations);
	// main logic
	Proposal bestsolution = fuse(generator,
							OccPen, SmoothPen, HOPen, segimages,
							SparsePen, SurfComplexityPen, winsize,
							dsi_left, maxdisp, scale,
							name, left, occ_left, dispfn);
	// display and store images
	IplImage *proposalimg = bestsolution.plot_proposal(scale);

	if (scale == 16.f)
		RoundDisp (proposalimg);

	cvvNamedWindow ("final disp", CV_WINDOW_AUTOSIZE);
	cvMoveWindow ("final disp", 2 * left->width, 0);
	cvShowImage ("final disp", proposalimg);
	
	cvSaveImage (dispfn, proposalimg);

	//cvWaitKey(0);
}