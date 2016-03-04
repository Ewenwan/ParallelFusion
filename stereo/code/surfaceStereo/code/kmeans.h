#include "opencv/cv.h"
#include "opencv/highgui.h"

#include <vector>
#include <algorithm>
using namespace std;

IplImage *kmeans_segmentation (IplImage *disp, int NumberOfClusters);