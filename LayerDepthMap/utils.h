//  utils.h
//  SurfaceStereo
//
//  Created by Chen Liu on 9/30/14.
//  Copyright (c) 2014 Chen Liu. All rights reserved.
//

#ifndef SurfaceStereo_utils_h
#define SurfaceStereo_utils_h

#include <vector>
#include <set>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

#include "Segment.h"


using namespace std;
using cv::Mat;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::VectorXd;
using Eigen::Vector3d;

//IplImage *refineDispImage(Mat &surface_id_image, const vector<double> &coordinates, Mat &disp_image, const int scale = 1);
vector<vector<double> > recoverSurfaceModels(Mat &surface_id_image, const vector<double> &point_cloud, Mat &disp_image, const bool use_3D_positions);
void normalizePointCloud(vector<double> &point_cloud);
vector<double> loadCoordinates(const char *filename);
void saveCoordinates(const vector<double> &coordinates, const char *filename);
Mat drawDispImage(const vector<vector<double> > &surface_models, const int width, const int height, const vector<double> &camera_parameters, const bool use_panoramic_image, const bool use_3D_positions);
Mat drawSurfaceIdImage(const Mat &surface_id_image, const int type = 0);
Mat blendImage(const Mat &image_1, const Mat &image_2, const int type = 0);
double calcFitError(vector<double> &surface_model, const double x, const double y, const double z, const int px, const int py);
//vector<double> fitLine2D(const vector<vector<double> > &points);
vector<double> calcCrossLine(const vector<double> &plane_1, const vector<double> &plane_2);
double calcDistance(vector<double> &surface_model, const int x, const int y, const int width, const int height);
void initCoordinates(vector<double> &coordinates, Mat &disp_image);
Mat readDispFromPFM(const char *filename, const int scale);
Mat deleteBoundary(const Mat &image, const int boundary_value);
vector<int> findPoints(const Mat &image, const int value);
void drawPoints(Mat &image, const vector<int> points, const int value);
Mat drawMask(const Mat &image, const set<int> indices);
void mergeSurfaces(Mat &image, const set<int> indices, const int value);
void drawSurface(Mat &image, const Mat &mask, const int ori_value, const int new_value);
Mat cropImage(const Mat &image, const int start_x, const int start_y, const int width, const int height);

//bool mergeTwoLines2D(const vector<long> position_pairs_1, const vector<long> position_pairs_2, const int width, const int height, vector<double> &new_line, vector<long> &new_line_position_pairs);
vector<int> deleteSmallSegments(const vector<int> &segmentation, const int width, const int small_segment_threshold);
Mat modifySurfaceIdImage(const Mat &surface_id_image, const int type);


void savePointCloudAsPly(const vector<double> &point_cloud, const char *filename);
void savePointCloudAsMesh(const vector<double> &point_cloud, const char *filename);
vector<int> loadSegmentation(const char *filename);
void saveSegmentation(const vector<int> &segmentation, const char *filename);
Mat drawSegmentationImage(const vector<int> &segmentation, const int width);
Mat drawSegmentationImage(const vector<int> &segmentation, const int width, const Mat &image, const char type);
vector<double> loadPointCloud(const char *filename);
void savePointCloud(const vector<double> &point_cloud, const char *filename);
Mat drawDispImage(const vector<double> &point_cloud, const int width, const MatrixXd &projection_matrix);
Mat drawDispImage(const vector<double> &point_cloud, const int width, const int height);


/*************Planes*************/
vector<double> fitPlane(const vector<double> &points, double &error_per_pixel);
vector<double> fitPlaneRobust(const vector<double> &points, const double plane_error_threshold);
vector<double> calcCenter(const vector<double> &points);
vector<double> calcRange(const vector<double> &points);
bool checkInRange(const vector<double> &point, const vector<double> &range);
vector<double> calcCameraParameters(const vector<double> &point_cloud, const int width, const bool use_panorama = false);
vector<double> calc3DPointOnPlane(const double image_x, const double image_y, const vector<double> &surface_model, const vector<double> &camera_parameters);
vector<double> calc3DPointOnImage(const double image_x, const double image_y, const vector<double> &camera_parameters);
double calcPlaneDistance(const double image_x, const double image_y, const vector<double> &surface_model_1, const vector<double> &surface_model_2, const vector<double> &camera_parameters);
map<int, map<int, int> > calcSurfaceRelations(const map<int, vector<double> > &point_clouds, const map<int, vector<double> > &surface_models, const double opposite_distance_threshold, const double num_outliers_threshold_ratio);
int findPlaneRelation(const vector<double> &points, const vector<double> &constraint_surface_model, const double opposite_distance_threshold, const double num_outliers_threshold_ratio);
bool checkRelationValid(const vector<double> &point, const vector<double> &constraint_surface_model, const int relation, const double opposite_distance_threshold);
vector<int> findOutmostSurfaces(const vector<vector<double> > &ranges, const vector<double> &extreme_values, const vector<vector<double> > &id_surface_models);
vector<double> calcExtremeValues(const vector<double> &point_cloud);

map<int, int> calcSurfaceObjectMap(const Mat &surface_id_image, const Mat &label_image);


//normalize point cloud
vector<double> normalizePointCloudByZ(const vector<double> &point_cloud);
//crop a region of image, point cloud and segmentation
void cropRegion(Mat &image, vector<double> &point_cloud, vector<int> &segmentation, const int start_x, const int start_y, const int new_width, const int new_height);
//zoom image, point cloud and segmentation
void zoomScene(Mat &image, vector<double> &point_cloud, const double scale_x, const double scale_y);
void drawMaskImage(const vector<bool> &mask, const int width, const int height, const string filename);

//get all combinations
vector<vector<int> > getCombinations(const vector<int> &candidates, const int num_elements);
//add an element multiple times
vector<vector<int> > fillWithNewElement(const vector<int> &current_values, const int new_element, const int num_elements);
//calculate distance to segment boundaries
vector<int> calcDistanceToBoundaries(const vector<int> segmentation, const int image_width, const int max_distance);

//draw an image based on an array
Mat drawArrayImage(const vector<double> &array, const int width, const int scale);

vector<double> smoothPointCloud(const vector<double> &point_cloud, const vector<int> &segmentation, const int image_width, const int image_height);

//generate a random probability
double randomProbability();

//calculate mean and svar
void calcStatistics(const vector<double> &values, double &mean, double &svar);

//calculate surface colors based on image and segmentation
map<int, int> calcSurfaceColors(const Mat &image, const vector<int> &segmentation);

//normalize values to [-range * svar, range * svar] / (range * svar) or get new value as (ori_value - mean) / svar
vector<double> normalizeValues(const vector<double> &values, const double range = -1);;

//draw a disparity image based on segmentation and surface_depths, and write it to filename
void writeSurfaceDepthsImage(const vector<int> &segmentation, const map<int, vector<double> > &surface_depths, const int image_width, const int image_height, const string filename);

//draw a disparity image based on segments, and write it to filename
void writeDispImageFromSegments(const vector<int> &labels, const int num_surfaces, const map<int, Segment> &segments, const int num_layers, const int image_width, const int image_height, const string filename);

//normalize a value to [0, 1] based on statistics
double normalizeStatistically(const double value, const double mean, const double svar, const double normalized_value_for_mean, const double scale_factor);

//calculate the number of bits with value 1
int calcNumOneBits(const int value);

//read point cloud from .obj file
vector<double> readPointCloudFromObj(const string filename, const int image_width, const int image_height, const double rotation_angle);

//calculate geodesic distance
double calcGeodesicDistance(const vector<vector<double> > &distance_map, const int width, const int height, const int start_pixel, const int end_pixel, const double distance_2D_weight);

//calculate geodesic distance for a set of pixels
vector<double> calcGeodesicDistances(const vector<vector<double> > &distance_map, const int width, const int height, const int start_pixel, const vector<int> end_pixels, const double distance_2D_weight);

struct PlaneFittingResidual {
PlaneFittingResidual(const double x, const double y, const double z) : x_(x), y_(y), z_(z) {};
    template <typename T>
    bool operator()(const T* const a, const T* const b, const T* const c, const T* const d, T* residual) const {
        residual[0] = (T)abs(a[0] * x_ + b[0] * y_ + c[0] * z_ - 1);
        return true;
    }
    
private:
    const double x_;
    const double y_;
    const double z_;
};

//flip a coin
inline bool flipCoin(const double head_possibility = 0.5)
{
  double random_value = static_cast<double>(rand()) / RAND_MAX;
  if (random_value < head_possibility)
    return true;
  else
    return false;
}

namespace Constants
{
  const double MAX_DEPTH_FOR_PENALTY = 0.3;
}

#endif
