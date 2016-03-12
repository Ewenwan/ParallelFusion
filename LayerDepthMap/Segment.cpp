#include "Segment.h"

#include <Eigen/Dense>

#include "utils.h"
#include "../base/cv_utils/cv_utils.h"

#include <iostream>

#include "TRW_S/MRFEnergy.h"
#include "BSplineSurface.h"


using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace Eigen;
using namespace cv_utils;


std::atomic<int> Segment::static_id(0);

Segment::Segment(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const vector<double> &camera_parameters, const vector<int> &pixels, const RepresenterPenalties &penalties, const DataStatistics &input_statistics, const int segment_type) : IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows), NUM_PIXELS_(image.cols * image.rows), CAMERA_PARAMETERS_(camera_parameters), penalties_(penalties), input_statistics_(input_statistics), segment_type_(segment_type), segment_id_(Segment::static_id++)
{
  if (segment_type == 0)
    //fitDispPlane(point_cloud, pixels, distance_to_boundaries);
    //fitDispPlaneRansac(point_cloud, normals, pixels);
    fitDepthPlane(image, point_cloud, normals, deleteInvalidPixels(point_cloud, pixels));
  else if (segment_type > 0) {
    if (pixels.size() > input_statistics_.bspline_surface_num_pixels_threshold) {
      segment_pixels_ = pixels;
      return;
    }
    fitBSplineSurface(image, point_cloud, normals, deleteInvalidPixels(point_cloud, pixels));
  }

  calcColorStatistics(image, segment_pixels_);
  calcSegmentMaskInfo();
  
  //calcConfidence();
}

Segment::Segment(const int image_width, const int image_height, const vector<double> &camera_parameters, const RepresenterPenalties &penalties, const DataStatistics &input_statistics) : IMAGE_WIDTH_(image_width), IMAGE_HEIGHT_(image_height), NUM_PIXELS_(image_width * image_height), CAMERA_PARAMETERS_(camera_parameters), penalties_(penalties), input_statistics_(input_statistics), segment_id_(Segment::static_id++)
{
}

Segment::Segment() : segment_id_(Segment::static_id++)
{
}


// void Segment::fitDispPlane(const std::vector<double> &point_cloud, const std::vector<int> &pixels, const std::vector<int> &distance_to_boundaries)
// {
//   if (pixels.size() < 3) {
//     plane_reliability_ = false;

//     disp_plane_ = vector<double>(3, 0);
//     segment_statistics_.disp_residual = 0;
//     if (pixels.size() > 0) {
//       double disp_sum = 0;
//       for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
//         disp_sum += 1 / point_cloud[*pixel_it * 3 + 2];
//       double disp_mean = disp_sum / pixels.size();
//       disp_plane_[2] = disp_mean;
//     }

//     disp_plane_[2] = round(disp_plane_[2] * 10000) / 10000;
//     calcDepthMap(point_cloud, pixels);
//     return;
//   }
  
//   plane_reliability_ = true;
  
//   map<int, vector<double> > segment_points;
//   for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
//     int x = *pixel_it % IMAGE_WIDTH_;
//     int y = *pixel_it / IMAGE_WIDTH_;
//     int distance = distance_to_boundaries.size() == 0 ? -1 : distance_to_boundaries[pixel_it - pixels.begin()];
//     //vector<double> point(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3);
//     vector<double> point(3);
//     point[0] = 1.0 * (x - IMAGE_WIDTH_ / 2) / IMAGE_WIDTH_;
//     point[1] = 1.0 * (y - IMAGE_HEIGHT_ / 2) / IMAGE_HEIGHT_;
//     point[2] = 1 / point_cloud[*pixel_it * 3 + 2];
//     segment_points[distance].insert(segment_points[distance].end(), point.begin(), point.end());
//   }
//   map<int, double> distance_weight_map;
//   distance_weight_map[-1] = 1;
//   for (int distance = 0; distance <= penalties_.segmentation_unconfident_region_width; distance++)
//     distance_weight_map[distance] = 1.0 * (distance + 1) / (penalties_.segmentation_unconfident_region_width + 2);

//   //double mean_x = -1, svar_x = -1, mean_y = -1, svar_y = -1, mean_z = -1, svar_z = -1;

//   vector<double> points;
//   vector<double> weights;
//   double weight_sum = 0;
//   for (map<int, vector<double> >::const_iterator distance_it = segment_points.begin(); distance_it != segment_points.end(); distance_it++) {
//     points.insert(points.end(), distance_it->second.begin(), distance_it->second.end());
//     weights.insert(weights.end(), distance_it->second.size() / 3, distance_weight_map[distance_it->first]);
//     weight_sum += distance_weight_map[distance_it->first] * distance_it->second.size() / 3;
//   }

//   const int NUM_POINTS = points.size() / 3;
  

//   // vector<double> disps(NUM_POINTS);
//   // for (int point_index = 0; point_index < NUM_POINTS; point_index++)
//   //   disps[point_index] = points[point_index * 3 + 2];
//   // double mean;
//   // calcStatistics(disps, mean, disp_svar_);
  
    
//   MatrixXd A(NUM_POINTS, 3);
//   VectorXd b(NUM_POINTS);
//   for (int point_index = 0; point_index < NUM_POINTS; point_index++) {
//     A(point_index, 0) = points[point_index * 3 + 0] * weights[point_index];
//     A(point_index, 1) = points[point_index * 3 + 1] * weights[point_index];
//     A(point_index, 2) = weights[point_index];
//     b(point_index) = points[point_index * 3 + 2] * weights[point_index];
//   }
//   VectorXd disp_plane = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

//   disp_plane_ = vector<double>(3);
//   for (int c = 0; c < 3; c++)
//     disp_plane_[c] = disp_plane(c);

//   for (int c = 0; c < 3; c++)
//     disp_plane_[c] = round(disp_plane_[c] * 10000) / 10000;
  
//   // disp_fitting_error_ = 0;
//   // for (int point_index = 0; point_index < NUM_POINTS; point_index++) {
//   //   double fitted_disp = points[point_index * 3 + 0] * disp_plane_[0] + points[point_index * 3 + 1] * disp_plane_[1] + disp_plane_[2];
//   //   double input_disp = points[point_index * 3 + 2];
//   //   disp_fitting_error_ += pow(fitted_disp - input_disp, 2);
//   // }
//   // disp_fitting_error_ = sqrt(disp_fitting_error_ / NUM_POINTS);
  
//   calcDepthMap(point_cloud, pixels);
// }

// void Segment::fitDispPlaneRansac(const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels)
// {
//   if (pixels.size() < 3) {
//     segment_type_ = -1;
//     segment_pixels_ = pixels;
    
//     disp_plane_ = vector<double>(3, 0);
//     depth_plane_ = vector<double>(4, 0);
//     // segment_statistics_.disp_residual = 0;
//     // segment_statistics_.disp_residual_svar = 0;
//     if (pixels.size() > 0) {
//       double disp_sum = 0;
//       double depth_sum = 0;
//       for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
//         disp_sum += 1 / point_cloud[*pixel_it * 3 + 2];
// 	depth_sum += point_cloud[*pixel_it * 3 + 2];
//       }
//       double disp_mean = disp_sum / pixels.size();
//       disp_plane_[2] = round(disp_mean * 10000) / 10000;
//       double depth_mean = depth_sum / pixels.size();
//       depth_plane_[2] = 1;
//       depth_plane_[3] = round(depth_mean * 10000) / 10000;
//     }

//     calcDepthMap(point_cloud, pixels);
//     return;
//   }

//   segment_type_ = 0;
  
//   vector<double> points;
//   for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
//     int x = *pixel_it % IMAGE_WIDTH_;
//     int y = *pixel_it / IMAGE_WIDTH_;
//     //vector<double> point(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3);
//     vector<double> point(3);
//     point[0] = 1.0 * (x - IMAGE_WIDTH_ / 2);
//     point[1] = 1.0 * (y - IMAGE_HEIGHT_ / 2);
//     point[2] = 1 / point_cloud[*pixel_it * 3 + 2];
//     points.insert(points.end(), point.begin(), point.end());
//   }

//   const int NUM_OUTER_ITERATIONS = 50;
//   const int NUM_INNER_ITERATIONS = 5;
  
//   int max_num_inliers = 0;
//   vector<double> max_num_inliers_disp_plane;
//   vector<double> max_num_inliers_effective_points;
//   vector<int> max_num_inliers_effective_pixels;
//   for (int outer_iteration = 0; outer_iteration < NUM_OUTER_ITERATIONS; outer_iteration++) {
//     set<int> initial_point_indices;
//     while (initial_point_indices.size() < 3)
//       initial_point_indices.insert(rand() % pixels.size());

//     vector<double> best_disp_plane;
//     vector<double> effective_points;
//     vector<int> effective_pixels;
//     for (set<int>::const_iterator point_index_it = initial_point_indices.begin(); point_index_it != initial_point_indices.end(); point_index_it++)
//       effective_points.insert(effective_points.end(), points.begin() + *point_index_it * 3, points.begin() + (*point_index_it + 1) * 3);
    
//     double disp_fitting_threshold = input_statistics_.disp_residual > 0 ? sqrt(input_statistics_.disp_residual) : 1000000;
//     vector<double> depth_plane;
//     for (int inner_iteration = 0; inner_iteration < NUM_INNER_ITERATIONS; inner_iteration++) {
//       const int NUM_POINTS = effective_points.size() / 3;
//       if (NUM_POINTS < 3)
// 	break;

//       MatrixXd A(NUM_POINTS, 3);
//       VectorXd b(NUM_POINTS);
//       for (int point_index = 0; point_index < NUM_POINTS; point_index++) {
// 	A(point_index, 0) = effective_points[point_index * 3 + 0];
// 	A(point_index, 1) = effective_points[point_index * 3 + 1];
// 	A(point_index, 2) = 1;
// 	b(point_index) = effective_points[point_index * 3 + 2];
//       }
//       VectorXd disp_plane = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
//       best_disp_plane = vector<double>(3);
//       for (int c = 0; c < 3; c++)
//         best_disp_plane[c] = disp_plane(c);

//       vector<double> points_3D;
//       for (vector<int>::const_iterator pixel_it = effective_pixels.begin(); pixel_it != effective_pixels.end(); pixel_it++)
//         points_3D.insert(points_3D.end(), point_cloud.begin() + *pixel_it * 3, point_cloud.begin() + (*pixel_it + 1) * 3);
//       vector<double> depth_plane = points_3D.size() >= 9 ? fitPlane(points_3D) : vector<double>();

//       double residual_sum = 0;
//       int num_fitted_pixels = 0;
//       vector<int> new_effective_pixels;
//       vector<double> new_effective_points;
//       for (int point_index = 0; point_index < points.size() / 3; point_index++) {
//         double fitted_disp = points[point_index * 3 + 0] * disp_plane(0) + points[point_index * 3 + 1] * disp_plane(1) + disp_plane(2);
//         double input_disp = points[point_index * 3 + 2];
// 	if (abs(fitted_disp - input_disp) > disp_fitting_threshold)
// 	  continue;

// 	vector<double> normal(normals.begin() + pixels[point_index] * 3, normals.begin() + (pixels[point_index] + 1) * 3);
// 	if (depth_plane.size() > 0) {
// 	  double cos_value = 0;
// 	  for (int c = 0; c < 3; c++)
// 	    cos_value += normal[c] * depth_plane[c];
// 	  double angle = acos(min(abs(cos_value), 1.0));
// 	  if (angle > input_statistics_.fitting_angle_threshold)
// 	    continue;
// 	}
	
// 	residual_sum += pow(fitted_disp  - input_disp, 2);
// 	num_fitted_pixels++;

// 	new_effective_points.insert(new_effective_points.end(), points.begin() + point_index * 3, points.begin() + (point_index + 1) * 3);
// 	new_effective_pixels.push_back(pixels[point_index]);
//       }

//       effective_points = new_effective_points;
//       effective_pixels = new_effective_pixels;

//       if (input_statistics_.disp_residual <= 0)
//         disp_fitting_threshold = min(disp_fitting_threshold, sqrt(residual_sum / num_fitted_pixels));

//       //cout << disp_fitting_threshold << '\t' << effective_points.size() / 3 << endl;
      
//       // effective_points.clear();
//       // effective_pixels.clear();
//       // for (int point_index = 0; point_index < points.size() / 3; point_index++) {
//       // 	double fitted_disp = points[point_index * 3 + 0] * disp_plane(0) + points[point_index * 3 + 1] * disp_plane(1) + disp_plane(2);
//       // 	double input_disp = points[point_index * 3 + 2];
//       // 	if (abs(fitted_disp  - input_disp) <= disp_fitting_threshold) {
//       // 	  effective_points.insert(effective_points.end(), points.begin() + point_index * 3, points.begin() + (point_index + 1) * 3);
//       // 	  effective_pixels.push_back(pixels[point_index]);
//       // 	}
//       // }
//     }

//     int num_inliers = effective_points.size() / 3;
//     //cout << num_inliers << '\t' << points.size() / 3 << endl;
//     if (num_inliers > max_num_inliers) {
//       max_num_inliers_disp_plane = best_disp_plane;
//       max_num_inliers = num_inliers;
//       max_num_inliers_effective_points = effective_points;
//       max_num_inliers_effective_pixels = effective_pixels;
//     }
//   }
//   //exit(1);
//   if (max_num_inliers < 3) {
//     segment_type_ = -1;
//     segment_pixels_ = pixels;
    
//     disp_plane_ = vector<double>(3, 0);
//     depth_plane_ = vector<double>(4, 0);
//     segment_statistics_.disp_residual = 0;
//     segment_statistics_.disp_residual_svar = 0;
//     if (pixels.size() > 0) {
//       double disp_sum = 0;
//       double depth_sum = 0;
//       for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
//         disp_sum += 1 / point_cloud[*pixel_it * 3 + 2];
//         depth_sum += point_cloud[*pixel_it * 3 + 2];
//       }
//       double disp_mean = disp_sum / pixels.size();
//       disp_plane_[2] = round(disp_mean * 10000) / 10000;
//       double depth_mean = depth_sum / pixels.size();
//       depth_plane_[2] = 1;
//       depth_plane_[3] = round(depth_mean * 10000) / 10000;
//     }

//     calcDepthMap(point_cloud, pixels);
//     return;
//   }


//   segment_pixels_ = max_num_inliers_effective_pixels;
  
//   disp_plane_ = vector<double>(3);
//   for (int c = 0; c < 3; c++)
//     disp_plane_[c] = round(max_num_inliers_disp_plane[c] * 10000) / 10000;

//   vector<double> points_3D;
//   for (vector<int>::const_iterator pixel_it = max_num_inliers_effective_pixels.begin(); pixel_it != max_num_inliers_effective_pixels.end(); pixel_it++)
//     points_3D.insert(points_3D.end(), point_cloud.begin() + *pixel_it * 3, point_cloud.begin() + (*pixel_it + 1) * 3);
//   depth_plane_ = fitPlane(points_3D);

  
//   double disp_residual_sum = 0;
//   double disp_residual2_sum = 0;
//   for (int point_index = 0; point_index < max_num_inliers_effective_points.size() / 3; point_index++) {
//     double fitted_disp = max_num_inliers_effective_points[point_index * 3 + 0] * disp_plane_[0] + max_num_inliers_effective_points[point_index * 3 + 1] * disp_plane_[1] + disp_plane_[2];
//     double input_disp = max_num_inliers_effective_points[point_index * 3 + 2];
//     double residual = pow(fitted_disp - input_disp, 2);
//     disp_residual_sum += residual;
//     disp_residual2_sum += pow(residual, 2);
//   }
//   segment_statistics_.disp_residual = disp_residual_sum / max_num_inliers;
//   segment_statistics_.disp_residual_svar = sqrt(disp_residual2_sum / max_num_inliers - pow(segment_statistics_.disp_residual, 2));
  
//   calcDepthMap(point_cloud, max_num_inliers_effective_pixels);
// }

// void Segment::fitDispPlaneRobustly(const std::vector<double> &point_cloud, const std::vector<int> &pixels)
// {
//   if (pixels.size() < 3) {
//     plane_reliability_ = false;

//     disp_plane_ = vector<double>(3, 0);
//     //disp_fitting_error_ = 0;
//     if (pixels.size() > 0) {
//       double depth_sum = 0;
//       for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
//         depth_sum += point_cloud[*pixel_it * 3 + 2];
//       double depth_mean = depth_sum / pixels.size();
//       if (depth_mean != 0)
//         disp_plane_[2] = 1.0 / depth_mean;
//     }

//     disp_plane_[2] = round(disp_plane_[2] * 10000) / 10000;
//     calcDepthMap(point_cloud, pixels);
//     return;
//   }

//   plane_reliability_ = true;
  
//   vector<double> points;
//   for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
//     int x = *pixel_it % IMAGE_WIDTH_;
//     int y = *pixel_it / IMAGE_WIDTH_;
//     //vector<double> point(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3);
//     vector<double> point(3);
//     point[0] = 1.0 * (x - IMAGE_WIDTH_ / 2) / IMAGE_WIDTH_;
//     point[1] = 1.0 * (y - IMAGE_HEIGHT_ / 2) / IMAGE_HEIGHT_;
//     point[2] = 1 / point_cloud[*pixel_it * 3 + 2];
//     points.insert(points.end(), point.begin(), point.end());
//   }

//   const int NUM_ITERATIONS = 10;
//   vector<double> effective_points = points;
//   for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
//     const int NUM_POINTS = effective_points.size() / 3;
//     if (NUM_POINTS < 3) {
//       plane_reliability_ = false;

//       disp_plane_ = vector<double>(3, 0);
//       //disp_fitting_error_ = 0;
//       double depth_sum = 0;
//       for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
//         depth_sum += point_cloud[*pixel_it * 3 + 2];
//       double depth_mean = depth_sum / pixels.size();
//       if (depth_mean != 0)
//         disp_plane_[2] = 1.0 / depth_mean;
      
//       disp_plane_[2] = round(disp_plane_[2] * 10000) / 10000;
//       calcDepthMap(point_cloud, pixels);
//       return;
//     }
//     MatrixXd A(NUM_POINTS, 3);
//     VectorXd b(NUM_POINTS);
//     for (int point_index = 0; point_index < NUM_POINTS; point_index++) {
//       A(point_index, 0) = effective_points[point_index * 3 + 0];
//       A(point_index, 1) = effective_points[point_index * 3 + 1];
//       A(point_index, 2) = 1;
//       b(point_index) = effective_points[point_index * 3 + 2];
//     }
//     VectorXd disp_plane = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
//     disp_plane_ = vector<double>(3);
//     for (int c = 0; c < 3; c++)
//       disp_plane_[c] = disp_plane(c);
    
//     effective_points.clear();
//     for (int point_index = 0; point_index < points.size() / 3; point_index++) {
//       double fitted_disp = points[point_index * 3 + 0] * disp_plane(0) + points[point_index * 3 + 1] * disp_plane(1) + disp_plane(2);
//       double input_disp = points[point_index * 3 + 2];
//       if (abs(1.0 / fitted_disp  - 1.0 / input_disp) <= penalties_.smoothness_depth_change_threshold)
//         effective_points.insert(effective_points.end(), points.begin() + point_index * 3, points.begin() + (point_index + 1) * 3);
//     }
//   }
  
//   for (int c = 0; c < 3; c++)
//     disp_plane_[c] = round(disp_plane_[c] * 10000) / 10000;
   
//   // disp_fitting_error_ = 0;
//   // for (int point_index = 0; point_index < points.size() / 3; point_index++) {
//   //   double fitted_disp = points[point_index * 3 + 0] * disp_plane_[0] + points[point_index * 3 + 1] * disp_plane_[1] + disp_plane_[2];
//   //   double input_disp = points[point_index * 3 + 2];
//   //   disp_fitting_error_ += pow(fitted_disp - input_disp, 2);
//   // }
//   // disp_fitting_error_ = sqrt(disp_fitting_error_ / (points.size() / 3));
  
//   calcDepthMap(point_cloud, pixels);
// }

// void Segment::fitDepthPlaneRansac(const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels)
// {
//   if (pixels.size() < 3) {
//     segment_type_ = -1;
//     segment_pixels_ = pixels;
    
//     disp_plane_ = vector<double>(3, 0);
//     depth_plane_ = vector<double>(4, 0);
//     // segment_statistics_.disp_residual = 0;
//     // segment_statistics_.disp_residual_svar = 0;
//     if (pixels.size() > 0) {
//       double disp_sum = 0;
//       double depth_sum = 0;
//       for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
//         disp_sum += 1 / point_cloud[*pixel_it * 3 + 2];
//         depth_sum += point_cloud[*pixel_it * 3 + 2];
//       }
//       double disp_mean = disp_sum / pixels.size();
//       disp_plane_[2] = round(disp_mean * 10000) / 10000;
//       double depth_mean = depth_sum / pixels.size();
//       depth_plane_[2] = 1;
//       depth_plane_[3] = round(depth_mean * 10000) / 10000;
//     }

//     calcDepthMap(point_cloud, pixels);
//     return;
//   }
//   segment_type_ = 0;
  
//   vector<double> point_cloud_range(6);
//   for (int c = 0; c < 3; c++) {
//     point_cloud_range[c * 2] = 1000000;
//     point_cloud_range[c * 2 + 1] = -1000000;
//   }
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     vector<double> point(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3);
//     for (int c = 0; c < 3; c++) {
//       if (point[c] < point_cloud_range[c * 2])
//         point_cloud_range[c * 2] = point[c];
//       if (point[c] > point_cloud_range[c * 2 + 1])
//         point_cloud_range[c * 2 + 1] = point[c];
//     }
//   }
      
//   const int NUM_OUTER_ITERATIONS = 50;
//   const int NUM_INNER_ITERATIONS = 5;
  
//   int max_num_inliers = 0;
//   vector<double> max_num_inliers_depth_plane;
//   vector<int> max_num_inliers_effective_pixels;
//   for (int outer_iteration = 0; outer_iteration < NUM_OUTER_ITERATIONS; outer_iteration++) {
//     set<int> initial_point_indices;
//     while (initial_point_indices.size() < 3)
//       initial_point_indices.insert(rand() % pixels.size());

//     vector<double> depth_plane;
//     vector<int> effective_pixels;
//     for (set<int>::const_iterator point_index_it = initial_point_indices.begin(); point_index_it != initial_point_indices.end(); point_index_it++)
//       effective_pixels.push_back(pixels[*point_index_it]);
    
//     double depth_fitting_threshold = input_statistics_.depth_residual > 0 ? sqrt(input_statistics_.depth_residual) : (point_cloud_range[5] - point_cloud_range[4]);
//     for (int inner_iteration = 0; inner_iteration < NUM_INNER_ITERATIONS; inner_iteration++) {
//       const int NUM_POINTS = effective_pixels.size();
//       if (NUM_POINTS < 3)
//         break;

//       vector<double> effective_points;
//       for (vector<int>::const_iterator pixel_it = effective_pixels.begin(); pixel_it != effective_pixels.end(); pixel_it++)
//         effective_points.insert(effective_points.end(), point_cloud.begin() + *pixel_it * 3, point_cloud.begin() + (*pixel_it + 1) * 3);
      
//       depth_plane = fitPlane(effective_points);
      
//       double residual_sum = 0;
//       int num_fitted_pixels = 0;
//       vector<int> new_effective_pixels;
//       for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
//         double fitted_depth = calcFittedDepth(depth_plane, *pixel_it);
//         double input_depth = point_cloud[*pixel_it * 3 + 2];
// 	// if (abs(fitted_depth - input_depth) > 0.1)
// 	//   cout << fitted_depth << '\t' << input_depth << endl;
// 	if (abs(fitted_depth - input_depth) > depth_fitting_threshold)
//           continue;

//         vector<double> normal(normals.begin() + *pixel_it * 3, normals.begin() + (*pixel_it + 1) * 3);
//         if (depth_plane.size() > 0) {
//           double cos_value = 0;
//           for (int c = 0; c < 3; c++)
//             cos_value += normal[c] * depth_plane[c];
//           double angle = acos(min(abs(cos_value), 1.0));
//           if (angle > input_statistics_.fitting_angle_threshold)
//             continue;
//         }
        
//         residual_sum += pow(fitted_depth  - input_depth, 2);
//         num_fitted_pixels++;

//         new_effective_pixels.push_back(*pixel_it);
//       }

//       effective_pixels = new_effective_pixels;
      
//       if (input_statistics_.depth_residual <= 0)
//         depth_fitting_threshold = min(depth_fitting_threshold, sqrt(residual_sum / num_fitted_pixels));
//     }

//     int num_inliers = effective_pixels.size();
//     //cout << num_inliers << '\t' << points.size() / 3 << endl;
//     if (num_inliers > max_num_inliers) {
//       max_num_inliers_depth_plane = depth_plane;
//       max_num_inliers = num_inliers;
//       max_num_inliers_effective_pixels = effective_pixels;
//     }
//   }
//   //exit(1);
  
//   if (max_num_inliers < 3) {
//     segment_type_ = -1;
//     segment_pixels_ = pixels;
    
//     disp_plane_ = vector<double>(3, 0);
//     depth_plane_ = vector<double>(4, 0);
//     segment_statistics_.disp_residual = 0;
//     segment_statistics_.disp_residual_svar = 0;
//     if (pixels.size() > 0) {
//       double disp_sum = 0;
//       double depth_sum = 0;
//       for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
//         disp_sum += 1 / point_cloud[*pixel_it * 3 + 2];
//         depth_sum += point_cloud[*pixel_it * 3 + 2];
//       }
//       double disp_mean = disp_sum / pixels.size();
//       disp_plane_[2] = round(disp_mean * 10000) / 10000;
//       double depth_mean = depth_sum / pixels.size();
//       depth_plane_[2] = 1;
//       depth_plane_[3] = round(depth_mean * 10000) / 10000;
//     }

//     calcDepthMap(point_cloud, pixels);
//     return;
//   }


//   segment_pixels_ = max_num_inliers_effective_pixels;
  
//   depth_plane_ = max_num_inliers_depth_plane;
//   disp_plane_ = vector<double>(3);
//   disp_plane_[0] = depth_plane_[0] / (CAMERA_PARAMETERS_[0] * depth_plane_[3]);
//   disp_plane_[1] = depth_plane_[1] / (CAMERA_PARAMETERS_[0] * depth_plane_[3]);
//   disp_plane_[2] = depth_plane_[2] / depth_plane_[3];
  
//   double disp_residual_sum = 0;
//   double disp_residual2_sum = 0;
//   for (vector<int>::const_iterator pixel_it = max_num_inliers_effective_pixels.begin(); pixel_it != max_num_inliers_effective_pixels.end(); pixel_it++) {
//     double fitted_depth = calcFittedDepth(max_num_inliers_depth_plane, *pixel_it);
//     double input_depth = point_cloud[*pixel_it * 3 + 2];
//     double residual = pow(1 / fitted_depth - 1 / input_depth, 2);
//     disp_residual_sum += residual;
//     disp_residual2_sum += pow(residual, 2);
//   }
//   segment_statistics_.disp_residual = disp_residual_sum / max_num_inliers;
//   segment_statistics_.disp_residual_svar = sqrt(disp_residual2_sum / max_num_inliers - pow(segment_statistics_.disp_residual, 2));
  
//   calcDepthMap(point_cloud, max_num_inliers_effective_pixels);
// }

void Segment::fitDepthPlane(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels)
{
  if (pixels.size() < 3) {
    fitParallelSurface(point_cloud, normals, pixels);
    return;
  }

  Mat blurred_image;
  GaussianBlur(image, blurred_image, cv::Size(3, 3), 0, 0);
  Mat blurred_hsv_image;
  blurred_image.convertTo(blurred_hsv_image, CV_32FC3, 1.0 / 255);
  cvtColor(blurred_hsv_image, blurred_hsv_image, CV_BGR2HSV);

  segment_type_ = 0;
  
  const int NUM_ITERATIONS = min(static_cast<int>(pixels.size() / 3), 300);
  
  int max_num_inliers = 0;
  vector<double> max_num_inliers_depth_plane;
  //Vec3f max_num_inliers_mean_color;
  for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
    set<int> initial_point_indices;
    while (initial_point_indices.size() < 3)
      initial_point_indices.insert(rand() % pixels.size());

    Vec3f sum_color(0, 0, 0);
    for (set<int>::const_iterator point_index_it = initial_point_indices.begin(); point_index_it != initial_point_indices.end(); point_index_it++)
      sum_color += blurred_hsv_image.at<Vec3f>(pixels[*point_index_it] / IMAGE_WIDTH_, pixels[*point_index_it] % IMAGE_WIDTH_);
    // Vec3f mean_color = sum_color / static_cast<int>(initial_point_indices.size());
    // bool color_consistency = true;
    // for (set<int>::const_iterator point_index_it = initial_point_indices.begin(); point_index_it != initial_point_indices.end(); point_index_it++)
    //   if (calcColorDiff(blurred_hsv_image.at<Vec3f>(pixels[*point_index_it] / IMAGE_WIDTH_, pixels[*point_index_it] % IMAGE_WIDTH_), mean_color) > input_statistics_.color_diff_threshold)
    //     color_consistency = false;
    // if (color_consistency == false)
    //   continue;
    
    vector<double> initial_points;
    for (set<int>::const_iterator point_index_it = initial_point_indices.begin(); point_index_it != initial_point_indices.end(); point_index_it++)
      initial_points.insert(initial_points.end(), point_cloud.begin() + pixels[*point_index_it] * 3, point_cloud.begin() + (pixels[*point_index_it] + 1) * 3);

    vector<double> depth_plane = fitPlane(initial_points);
    if (depth_plane.size() == 0)
      continue;
    
    int num_inliers = 0;
    for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
      // double fitted_depth = calcFittedDepth(depth_plane, *pixel_it);
      // double input_depth = point_cloud[*pixel_it * 3 + 2];
        // if (abs(fitted_depth - input_depth) > 0.1)
        //   cout << fitted_depth << '\t' << input_depth << endl;
        // if (abs(fitted_depth - input_depth) > depth_fitting_threshold)
        //   continue;
      assert(point_cloud[*pixel_it * 3 + 2] > 0);
      
      if (point_cloud[*pixel_it * 3 + 2] < 0)
	continue;
      vector<double> point(point_cloud.begin() + *pixel_it * 3, point_cloud.begin() + (*pixel_it + 1) * 3);
      double distance = depth_plane[3];
      for (int c = 0; c < 3; c++)
	distance -= depth_plane[c] * point[c];
      distance = abs(distance);
      if (distance > input_statistics_.pixel_fitting_distance_threshold)
	continue;
      
      vector<double> normal(normals.begin() + *pixel_it * 3, normals.begin() + (*pixel_it + 1) * 3);
      double cos_value = 0;
      for (int c = 0; c < 3; c++)
      	cos_value += normal[c] * depth_plane[c];
      double angle = acos(min(abs(cos_value), 1.0));
      if (sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2)) < 0.000001)
        angle = 0;
      if (angle > input_statistics_.pixel_fitting_angle_threshold)
      	continue;

      // Vec3f color = blurred_hsv_image.at<Vec3f>(*pixel_it / IMAGE_WIDTH_, *pixel_it % IMAGE_WIDTH_);
      // if (calcColorDiff(color, mean_color) > input_statistics_.color_diff_threshold)
      // 	continue;
      
      num_inliers++;
    }
    //cout << num_inliers << '\t' << points.size() / 3 << endl;
    if (num_inliers > max_num_inliers) {
      max_num_inliers_depth_plane = depth_plane;
      //max_num_inliers_mean_color = mean_color;
      max_num_inliers = num_inliers;
    }
  }
  //exit(1);
  
  if (max_num_inliers < 3) {
    fitParallelSurface(point_cloud, normals, pixels);
    return;
  }

  depth_plane_ = max_num_inliers_depth_plane;
  segment_pixels_.clear();
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
    vector<double> point(point_cloud.begin() + *pixel_it * 3, point_cloud.begin() + (*pixel_it + 1) * 3);
    double distance = depth_plane_[3];
    for (int c = 0; c < 3; c++)
      distance -= depth_plane_[c] * point[c];
    distance = abs(distance);
    if (distance > input_statistics_.pixel_fitting_distance_threshold)
      continue;
      
    vector<double> normal(normals.begin() + *pixel_it * 3, normals.begin() + (*pixel_it + 1) * 3);
    double cos_value = 0;
    for (int c = 0; c < 3; c++)
      cos_value += normal[c] * depth_plane_[c];
    double angle = acos(min(abs(cos_value), 1.0));
    if (sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2)) < 0.000001)
      angle = 0;
    if (angle > input_statistics_.pixel_fitting_angle_threshold)
      continue;

    // Vec3f color = blurred_hsv_image.at<Vec3f>(*pixel_it / IMAGE_WIDTH_, *pixel_it % IMAGE_WIDTH_);
    // if (calcColorDiff(color, max_num_inliers_mean_color) > input_statistics_.color_diff_threshold)
    //   continue;
    
    segment_pixels_.push_back(*pixel_it);
  }
  
  segment_pixels_ = findLargestConnectedComponent(point_cloud, segment_pixels_);
  if (segment_pixels_.size() < 3) {
    fitParallelSurface(point_cloud, normals, pixels);
    return;
  }
  
  vector<double> fitted_points;
  for (vector<int>::const_iterator pixel_it = segment_pixels_.begin(); pixel_it != segment_pixels_.end(); pixel_it++) {
    vector<double> point(point_cloud.begin() + *pixel_it * 3, point_cloud.begin() + (*pixel_it + 1) * 3);
    fitted_points.insert(fitted_points.end(), point.begin(), point.end());
  }
  depth_plane_ = fitPlane(fitted_points);
  
  // static int test_count = 0;
  // stringstream filename;
  // filename << "Test/plane_" << test_count << ".ply";
  // test_count++;
  // savePointCloudAsPly(fitted_points, filename.str().c_str());


  
  // double disp_residual_sum = 0;
  // double disp_residual2_sum = 0;
  // for (vector<int>::const_iterator pixel_it = max_num_inliers_effective_pixels.begin(); pixel_it != max_num_inliers_effective_pixels.end(); pixel_it++) {
  //   double fitted_depth = calcFittedDepth(max_num_inliers_depth_plane, *pixel_it);
  //   double input_depth = point_cloud[*pixel_it * 3 + 2];
  //   double residual = pow(1 / fitted_depth - 1 / input_depth, 2);
  //   disp_residual_sum += residual;
  //   disp_residual2_sum += pow(residual, 2);
  // }
  // segment_statistics_.disp_residual = disp_residual_sum / max_num_inliers;
  // segment_statistics_.disp_residual_svar = sqrt(disp_residual2_sum / max_num_inliers - pow(segment_statistics_.disp_residual, 2));
  
  calcDepthMap(point_cloud, segment_pixels_);
}

void Segment::fitParallelSurface(const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels)
{
  //assert(false);
  //cout << "fit parallel surface" << endl;
  segment_type_ = -1;
    
  disp_plane_ = vector<double>(3, 0);
  depth_plane_ = vector<double>(4, 0);
  // segment_statistics_.disp_residual = 0;
  // segment_statistics_.disp_residual_svar = 0;
  if (pixels.size() > 0) {
    double disp_sum = 0;
    double depth_sum = 0;
    for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
      //cout << *pixel_it << endl;
      if (point_cloud[*pixel_it * 3 + 2] < 0)
	continue;
      disp_sum += 1 / point_cloud[*pixel_it * 3 + 2];
      depth_sum += point_cloud[*pixel_it * 3 + 2];
      segment_pixels_.push_back(*pixel_it);
    }
    double disp_mean = disp_sum / segment_pixels_.size();
    disp_plane_[2] = round(disp_mean * 10000) / 10000;
    double depth_mean = depth_sum / segment_pixels_.size();
    depth_plane_[2] = 1;
    depth_plane_[3] = round(depth_mean * 10000) / 10000;
  }
  
  calcDepthMap(point_cloud, pixels);
}

void Segment::calcColorStatistics(const Mat &image, const vector<int> &pixels)
{
  if (pixels.size() < 3) {
    // segment_statistics_.color_likelihood = 0;
    // segment_statistics_.color_likelihood_svar = 0;
    return;
  }
  Mat segment_samples(pixels.size(), 2, CV_32FC1);
  Mat blurred_image;
  GaussianBlur(image, blurred_image, cv::Size(3, 3), 0, 0);
  Mat blurred_hsv_image;
  blurred_image.convertTo(blurred_hsv_image, CV_32FC3, 1.0 / 255);
  cvtColor(blurred_hsv_image, blurred_hsv_image, CV_BGR2HSV);
  
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
    Vec3f color = blurred_hsv_image.at<Vec3f>(*pixel_it / IMAGE_WIDTH_, *pixel_it % IMAGE_WIDTH_);
    segment_samples.at<float>(pixel_it - pixels.begin(), 0) = color[1] * cos(color[0] * M_PI / 180);
    segment_samples.at<float>(pixel_it - pixels.begin(), 1) = color[1] * sin(color[0] * M_PI / 180);
    //segment_samples.at<float>(pixel_it - pixels.begin(), 2) = color[2] * 0.1;
    
    
    // Vec3b color = image.at<Vec3b>(*pixel_it / IMAGE_WIDTH_, *pixel_it % IMAGE_WIDTH_);
    // for (int c = 0; c < 3; c++)
    //   segment_samples.at<float>(pixel_it - pixels.begin(), c) = 1.0 * color[c] / 256;
    
    // segment_samples.at<float>(pixel_it - pixels.begin(), 3) = 1.0 * (*pixel_it % IMAGE_WIDTH_) / IMAGE_WIDTH_;
    // segment_samples.at<float>(pixel_it - pixels.begin(), 4) = 1.0 * (*pixel_it / IMAGE_WIDTH_) / IMAGE_HEIGHT_;
  }
  
  GMM_ = EM::create();
  GMM_->setClustersNumber(2);
  Mat log_likelihoods(pixels.size(), 1, CV_64FC1);
  GMM_->trainEM(segment_samples, log_likelihoods, noArray(), noArray());
  double likelihood_sum = 0;
  double likelihood2_sum = 0;
  for (int i = 0; i < pixels.size(); i++) {
    double likelihood = log_likelihoods.at<double>(i, 0);
    likelihood_sum += likelihood;
    //cout << likelihood << endl;
    likelihood2_sum += pow(likelihood, 2);
  }
  // segment_statistics_.color_likelihood = likelihood_sum / pixels.size();
  // segment_statistics_.color_likelihood_svar = sqrt(max(likelihood2_sum / pixels.size() - pow(segment_statistics_.color_likelihood, 2), 0.0));
  
  //cout << "likelihood: " << color_likelihood_ << endl;
  
  
  // map<int, vector<double> > segment_colors;
  // for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
  //   int x = *pixel_it % IMAGE_WIDTH_;
  //   int y = *pixel_it / IMAGE_WIDTH_;
  //   int distance = distance_to_boundaries.size() == 0 ? -1 : distance_to_boundaries[pixel_it - pixels.begin()];
  //   //vector<double> point(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3);
  //   Vec3b color = image.at<Vec3b>(y, x);
  //   for (int c = 0; c < 3; c++)
  //     segment_colors[distance].push_back(color[c]);
  // }
  // map<int, double> distance_weight_map;
  // distance_weight_map[-1] = 1;
  // for (int distance = 0; distance <= penalties_.segmentation_unconfident_region_width; distance++)
  //   distance_weight_map[distance] = 1.0 * (distance + 1) / (penalties_.segmentation_unconfident_region_width + 2);

  // //double mean_x = -1, svar_x = -1, mean_y = -1, svar_y = -1, mean_z = -1, svar_z = -1;

  // vector<double> colors;
  // vector<double> weights;
  // double weight_sum = 0;
  // for (map<int, vector<double> >::const_iterator distance_it = segment_colors.begin(); distance_it != segment_colors.end(); distance_it++) {
  //   colors.insert(colors.end(), distance_it->second.begin(), distance_it->second.end());
  //   weights.insert(weights.end(), distance_it->second.size() / 3, distance_weight_map[distance_it->first]);
  //   weight_sum += distance_weight_map[distance_it->first] * distance_it->second.size() / 3;
  // }

  // const int NUM_POINTS = pixels.size();
  // for (vector<double>::iterator weight_it = weights.begin(); weight_it != weights.end(); weight_it++)
  //   *weight_it /= weight_sum;
  
  // vector<double> rgb_sum(3, 0);
  // vector<double> rgb_sum2(3, 0);
  // for (int point_index = 0; point_index < NUM_POINTS; point_index++) {
  //   for (int c = 0; c < 3; c++) {
  //     rgb_sum[c] += colors[point_index * 3 + c] * weights[point_index];
  //     rgb_sum2[c] += pow(colors[point_index * 3 + c], 2) * weights[point_index];
  //   }
  // }

  // rgb_mean_ = rgb_sum;
  // rgb_svar_.assign(3, 0);
  // for (int c = 0; c < 3; c++)
  //   rgb_svar_[c] = sqrt(rgb_sum2[c] - pow(rgb_sum[c], 2));
}

void Segment::calcDepthMap(const vector<double> &point_cloud, const vector<int> &fitted_pixels)
{
  //cout << depth_plane_.size() << '\t' << depth_plane_[0] << '\t' << depth_plane_[1] << '\t' << depth_plane_[2] << '\t' << depth_plane_[3] << endl;
  // for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
  //   cout << *pixel_it << endl;
  disp_plane_ = vector<double>(3);
  disp_plane_[0] = depth_plane_[0] / (CAMERA_PARAMETERS_[0] * depth_plane_[3]);
  disp_plane_[1] = depth_plane_[1] / (CAMERA_PARAMETERS_[0] * depth_plane_[3]);
  disp_plane_[2] = depth_plane_[2] / depth_plane_[3];

  depth_map_ = vector<double>(IMAGE_WIDTH_ * IMAGE_HEIGHT_);
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    double u = pixel % IMAGE_WIDTH_ - CAMERA_PARAMETERS_[1];
    double v = pixel / IMAGE_WIDTH_ - CAMERA_PARAMETERS_[2];
    //double depth = 1 / ((plane(0) * u + plane(1) * v + plane(2)) / plane(3));
    
    double disp = disp_plane_[0] * u + disp_plane_[1] * v + disp_plane_[2];
    double depth = disp != 0 ? 1 / disp : 0;
    // if (depth <= 0)
    //   depth = -1;
    if (depth > 10)
      depth = 10;
    depth_map_[pixel] = depth;
  }

  // if (fitted_pixels.size() > 0) {
  //   double depth_residual_sum = 0;
  //   double depth_residual2_sum = 0;
  //   for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++) {
  //     double input_depth = point_cloud[*pixel_it * 3 + 2];
  //     double fitted_depth = depth_map_[*pixel_it];
  //     double depth_residual = pow(fitted_depth - input_depth, 2);
  //     depth_residual_sum += depth_residual;
  //     depth_residual2_sum += pow(depth_residual, 2);
  //   }
  //   segment_statistics_.depth_residual = depth_residual_sum / fitted_pixels.size();
  //   segment_statistics_.depth_residual_svar = sqrt(depth_residual2_sum / fitted_pixels.size() - pow(segment_statistics_.depth_residual, 2));
  // }
}

vector<double> Segment::getDepthMap() const
{
  return depth_map_;
}

double Segment::getDepth(const int pixel) const
{
  return depth_map_[pixel];
}

double Segment::getDepth(const double x_ratio, const double y_ratio) const
{
  double x = IMAGE_WIDTH_ * x_ratio;
  double y = IMAGE_HEIGHT_ * y_ratio;
  int lower_x = max(static_cast<int>(floor(x)), 0);
  int upper_x = min(static_cast<int>(ceil(x)), IMAGE_WIDTH_ - 1);
  int lower_y = max(static_cast<int>(floor(y)), 0);
  int upper_y = min(static_cast<int>(ceil(y)), IMAGE_HEIGHT_ - 1);
  if (lower_x == upper_x && lower_y == upper_y)
    return depth_map_[lower_y * IMAGE_WIDTH_ + lower_x];
  else if (lower_x == upper_x)
    return depth_map_[lower_y * IMAGE_WIDTH_ + lower_x] * (upper_y - y) + depth_map_[upper_y * IMAGE_WIDTH_ + lower_x] * (y - lower_y);
  else if (lower_y == upper_y)
    return depth_map_[lower_y * IMAGE_WIDTH_ + lower_x] * (upper_x - x) + depth_map_[lower_y * IMAGE_WIDTH_ + upper_x] * (x - lower_x);
  else {
    double area_1 = (x - lower_x) * (y - lower_y);
    double area_2 = (x - lower_x) * (upper_y - y);
    double area_3 = (upper_x - x) * (y - lower_y);
    double area_4 = (upper_x - x) * (upper_y - y);
    double depth_1 = depth_map_[lower_y * IMAGE_WIDTH_ + lower_x];
    double depth_2 = depth_map_[upper_y * IMAGE_WIDTH_ + lower_x];
    double depth_3 = depth_map_[lower_y * IMAGE_WIDTH_ + upper_x];
    double depth_4 = depth_map_[upper_y * IMAGE_WIDTH_ + upper_x];

    return depth_1 * area_4 + depth_2 * area_3 + depth_3 * area_2 + depth_4 * area_1;
  }
}

vector<double> Segment::getDepthPlane() const
{
  return depth_plane_;
}

double Segment::getConfidence() const
{
  return 1;
  return segment_confidence_;
}

int Segment::getType() const
{
  return segment_type_;
}

void Segment::calcConfidence()
{
  // int num_overlapping_visible_pixels = 0;
  // vector<bool> segmentation_pixel_mask(NUM_PIXELS_, false);
  // for (vector<int>::const_iterator pixel_it = segmentation_pixels_.begin(); pixel_it != segmentation_pixels_.end(); pixel_it++)
  //   segmentation_pixel_mask[*pixel_it] = true;
  // for (vector<int>::const_iterator pixel_it = visible_pixels.begin(); pixel_it != visible_pixels.end(); pixel_it++)
  //   if (segmentation_pixel_mask[*pixel_it] == true)
  //     num_overlapping_visible_pixels++;

  
  // double color_confidence = 1;
  // for (int c = 0; c < 3; c++)
  //   color_confidence *= exp(-pow(rgb_svar_[c] / 50, 2) / 2);

  
  //cout << color_confidence << endl;

  //cout << disp_fitting_error_ << '\t' << statistics_.disp_svar << endl;

  segment_confidence_ = 1;
  //segment_confidence_ = (1 - normalizeStatistically(segment_statistics_.depth_residual, input_statistics_.depth_residual, input_statistics_.depth_residual_svar, 0.2, 0.5)) * normalizeStatistically(segment_statistics_.color_likelihood, input_statistics_.color_likelihood, input_statistics_.color_likelihood_svar, 0.8, 0.5);
  if (segment_type_ == -1)
    segment_confidence_ = 0;
}

void Segment::calcSegmentMaskInfo()
{
  segment_mask_.assign(NUM_PIXELS_, false);
  int min_x = IMAGE_WIDTH_;
  int max_x = -1;
  int min_y = IMAGE_HEIGHT_;
  int max_y = -1;
  double sum_x = 0;
  double sum_y = 0;
  for (vector<int>::const_iterator pixel_it = segment_pixels_.begin(); pixel_it != segment_pixels_.end(); pixel_it++) {
    segment_mask_[*pixel_it] = true;
    int x = *pixel_it % IMAGE_WIDTH_;
    int y = *pixel_it / IMAGE_WIDTH_;
    if (x < min_x)
      min_x = x;
    if (x > max_x)
      max_x = x;
    if (y < min_y)
      min_y = y;
    if (y > max_y)
      max_y = y;
    sum_x += x;
    sum_y += y;
  }
  //segment_radius_ = sqrt((max_x - min_x + 1) * (max_y - min_y + 1));
  segment_radius_ = sqrt(segment_pixels_.size()) * 3;
  segment_center_x_ = sum_x / segment_pixels_.size();
  segment_center_y_ = sum_y / segment_pixels_.size();

  calcDistanceMap();
}

// double Segment::calcDistanceRatio2D(const int pixel)
// {
//   if (segment_mask_[pixel] == true)
//     return 0;
//   double distance = sqrt(pow(pixel % IMAGE_WIDTH_ - segment_center_x_, 2) + pow(pixel / IMAGE_WIDTH_ - segment_center_y_, 2));
//   return max(distance / segment_radius_, 1.0);
// }

void Segment::writeSegmentImage(const string filename)
{
  Mat segment_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  for (vector<int>::const_iterator pixel_it = segment_pixels_.begin(); pixel_it != segment_pixels_.end(); pixel_it++)
    segment_image.at<uchar>(*pixel_it / IMAGE_WIDTH_, *pixel_it % IMAGE_WIDTH_) = 255;
  imwrite(filename, segment_image);
}

Segment &Segment::operator =(const Segment &segment)
{
  IMAGE_WIDTH_ = segment.IMAGE_WIDTH_;
  IMAGE_HEIGHT_ = segment.IMAGE_HEIGHT_;
  NUM_PIXELS_ = segment.NUM_PIXELS_;
  CAMERA_PARAMETERS_ = segment.CAMERA_PARAMETERS_;
  penalties_ = segment.penalties_;
  segment_pixels_ = segment.segment_pixels_;
  segment_mask_ = segment.segment_mask_;
  segment_radius_ = segment.segment_radius_;
  segment_center_x_ = segment.segment_center_x_;
  segment_center_y_ = segment.segment_center_y_;
  distance_map_ = segment.distance_map_;
  segment_type_ = segment.segment_type_;
  disp_plane_ = segment.disp_plane_;
  depth_plane_ = segment.depth_plane_;
  input_statistics_ = segment.input_statistics_;
  //segment_statistics_ = segment.segment_statistics_;
  depth_map_ = segment.depth_map_;
  normals_ = segment.normals_;
  GMM_ = segment.GMM_;
  segment_confidence_ = segment.segment_confidence_;

  segment_id_ = segment.segment_id_;
  
  // Mat sample(1, 5, CV_32FC1);
  // for (int c = 0; c < 5; c++)
  //   sample.at<float>(0, c) = rand() % 256;
  // Vec2d result_1 = segment.GMM_->predict2(sample, noArray());
  // Vec2d result_2 = GMM_->predict2(sample, noArray());
  // cout << result_1[0] << '\t' << result_2[0] << endl;
  
  return *this;
}

ostream & operator <<(ostream &out_str, const Segment &segment)
{
  out_str << segment.segment_type_ << endl;
  out_str << segment.segment_pixels_.size() << endl;
  for (vector<int>::const_iterator pixel_it = segment.segment_pixels_.begin(); pixel_it != segment.segment_pixels_.end(); pixel_it++)
    out_str << *pixel_it << '\t';
  out_str << endl;
  //out_str << segment.visible_pixels_.size() << endl;
  // for (vector<int>::const_iterator pixel_it = segment.visible_pixels_.begin(); pixel_it != segment.visible_pixels_.end(); pixel_it++)
  //   out_str << *pixel_it << '\t';
  // out_str << endl;
  // for (int c = 0; c < 3; c++)
  //   out_str << segment.disp_plane_[c] << '\t';
  // out_str << endl;
  if (segment.segment_type_ == 0) {
    for (int c = 0; c < 4; c++)
      out_str << segment.depth_plane_[c] << '\t';
    out_str << endl;
  } else if (segment.segment_type_ > 0) {
    for (vector<double>::const_iterator depth_it = segment.depth_map_.begin(); depth_it != segment.depth_map_.end(); depth_it++)
      out_str << *depth_it << '\t';
    out_str << endl;
  }
  //out_str << segment.segment_statistics_;
  //out_str << segment.segment_confidence_ << endl;
  return out_str;
}

istream & operator >>(istream &in_str, Segment &segment)
{
  in_str >> segment.segment_type_;
  int num_segment_pixels;
  in_str >> num_segment_pixels;
  segment.segment_pixels_ = vector<int>(num_segment_pixels);
  for (int pixel_index = 0; pixel_index < num_segment_pixels; pixel_index++)
    in_str >> segment.segment_pixels_[pixel_index];
  segment.calcSegmentMaskInfo();
  // int num_visible_pixels;
  // in_str >> num_visible_pixels; 
  // segment.visible_pixels_ = vector<int>(num_visible_pixels);
  // for (int pixel_index = 0; pixel_index < num_visible_pixels; pixel_index++)
  //   in_str >> segment.visible_pixels_[pixel_index];
  
  // segment.disp_plane_.assign(3, 0);
  // for (int c = 0; c < 3; c++)
  //   in_str >> segment.disp_plane_[c];
  if (segment.segment_type_ == 0) {
    segment.depth_plane_.assign(4, 0);
    for (int c = 0; c < 4; c++)
      in_str >> segment.depth_plane_[c];
    segment.calcDepthMap();
  } else if (segment.segment_type_ > 0) {
    vector<double> depth_map(segment.NUM_PIXELS_, 0);
    for (int pixel = 0; pixel < segment.NUM_PIXELS_; pixel++)
      in_str >> depth_map[pixel];
    segment.depth_map_ = depth_map;
    segment.normals_ = calcNormals(segment.calcPointCloud(), segment.IMAGE_WIDTH_, segment.IMAGE_HEIGHT_);
  }
  //in_str >> segment.segment_statistics_;
  //in_str >> segment.segment_confidence_;
  //  segment.calcConfidence();
  return in_str;
}

double Segment::predictColorLikelihood(const int pixel, const Vec3f hsv_color) const
{
  if (segment_type_ == -1)
    return 0;
  Mat sample(1, 2, CV_64FC1);
  sample.at<double>(0, 0) = hsv_color[1] * cos(hsv_color[0] * M_PI / 180);
  sample.at<double>(0, 1) = hsv_color[1] * sin(hsv_color[0] * M_PI / 180);
  //sample.at<double>(0, 2) = hsv_color[2] * 0.1;
  
    
  // for (int c = 0; c < 3; c++)
  //   sample.at<double>(0, c) = 1.0 * color[c] / 256;
  
  // sample.at<double>(0, 3) = 1.0 * (pixel % IMAGE_WIDTH_) / IMAGE_WIDTH_;
  // sample.at<double>(0, 4) = 1.0 * (pixel / IMAGE_WIDTH_) / IMAGE_HEIGHT_;

  
  //cout << sample << endl;
  
  // Mat means = GMM_->getMeans();
  // const int NUM_CLUSTERS = means.rows;
  // const int NUM_DIMENSIONS = means.cols;
  // vector<Mat> covs;
  // GMM_->getCovs(covs);
  // double max_probability = -1;
  // double probability_sum = 0;
  // int max_probability_cluster_index = -1;
  // for (int cluster_index = 0; cluster_index < NUM_CLUSTERS; cluster_index++) {
  //   Mat mean = means.row(cluster_index);
  //   Mat cov = covs[cluster_index];
  //   Mat cov_inv;
  //   invert(cov, cov_inv);
  //   Mat diff_transpose;
  //   transpose(sample - mean, diff_transpose);
  //   Mat multiply_result = (sample - mean) * cov_inv * diff_transpose;
  //   double probability = 1 / sqrt(pow(2 * M_PI, NUM_DIMENSIONS) * determinant(cov)) * exp(-0.5 * multiply_result.at<double>(0, 0));
  //   if (probability > max_probability) {
  //     max_probability_cluster_index = cluster_index;
  //     max_probability = probability;
  //   }
  //   break;
  // }
  // cout << max_probability << endl;
  // return max_probability;


  Vec2d prediction = GMM_->predict2(sample, noArray());
  Mat weights = GMM_->getWeights();
  return prediction[0] + log(weights.at<double>(0, prediction[1]));
  // Mat mean = GMM_->getMeans().row(prediction[1]);
  // Vec2d mean_prediction = GMM_->predict2(mean, noArray());
  // double color_likelihood = pow(2, prediction[0] - mean_prediction[0]);
    
  // if (isnan(float(color_likelihood)))
  //   cout << sample << endl; //'\t' << mean << '\t' << prediction << '\t' << prediction << '\t' << 
  //  color_likelihood << endl;

  // if (color_likelihood > 1) {
  //   color_likelihood = 1;
  //   // cout << sample << endl;
  //   // cout << prediction << endl;
  //   // cout << mean << endl;
  //   // cout << mean_prediction << endl;

  //   // vector<Mat> covs;
  //   // GMM_->getCovs(covs);
  //   // Mat cov = covs[mean_prediction[1]];
  //   // Mat cov_inv;
  //   // invert(cov, cov_inv);
  //   // Mat diff_transpose;
  //   // transpose(sample - mean, diff_transpose);
  //   // Mat multiply_result = (sample - mean) * cov_inv * diff_transpose;
  //   // double probability = 1 / sqrt(pow(2 * M_PI, 3) * determinant(cov)) * exp(-0.5 * multiply_result.at<double>(0, 0));
  //   // cout << GMM_->getWeights() << endl;
  //   // cout << probability << endl;
  //   // exit(1);
  // }
  
  //double color_likelihood = normalizeStatistically(result[0], input_statistics_.color_likelihood, input_statistics_.color_likelihood_svar, 1, 0.4);

  // if (pixel == 0) {
  // vector<Mat> covs;
  // GMM_->getCovs(covs);
  // Mat cov = covs[mean_prediction[1]];
  // cout << cov << endl;
  // cout << sample << '\t' << mean << '\t' << color_likelihood << endl;
  // exit(1);
  //cout << GMM_->getWeights() << endl;
  //    cout << result << endl;
  //   exit(1);
  // }
  
  //return color_likelihood;
}

void Segment::setGMM(const Ptr<EM> GMM)
{
  GMM_ = GMM;
}

void Segment::setGMM(const cv::FileNode GMM_file_node)
{
  GMM_ = EM::create();
  GMM_->read(GMM_file_node);
}

Ptr<EM> Segment::getGMM() const
{
  return GMM_;
}

DataStatistics Segment::getSegmentStatistics() const
{
  return segment_statistics_;
}

int Segment::getNumSegmentPixels() const
{
  return segment_pixels_.size();
}

vector<int> Segment::getSegmentPixels() const
{
  return segment_pixels_;
}

bool Segment::checkPixelFitting(const Mat &hsv_image, const vector<double> &point_cloud, const vector<double> &normals, const int pixel) const
{
  // if (use_sub_segment_ && segment_mask_[pixel])
  //   return true;
  
  if (segment_type_ == -1)
    return false;

  if (depth_map_[pixel] < 0)
    return false;

  Vec3f color = hsv_image.at<Vec3f>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
  if (predictColorLikelihood(pixel, color) < input_statistics_.pixel_fitting_color_likelihood_threshold) {
    return false;
  }
  
  if (segment_type_ > 0) {
    double distance = abs(depth_map_[pixel] - point_cloud[pixel * 3 + 2]);
    if (point_cloud[pixel * 3 + 2] < 0)
      distance = 0;
    if (distance > input_statistics_.pixel_fitting_distance_threshold)
      return false;
    
    vector<double> normal(normals.begin() + pixel * 3, normals.begin() + (pixel + 1) * 3);
    vector<double> surface_normal(normals_.begin() + pixel * 3, normals_.begin() + (pixel + 1) * 3);
    double cos_value = 0;
    for (int c = 0; c < 3; c++)
      cos_value += normal[c] * surface_normal[c];
    double angle = acos(min(abs(cos_value), 1.0));
    if (sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2)) < 0.000001)
      angle = 0;
    if (angle > input_statistics_.pixel_fitting_angle_threshold)
      return false;
    
    return true;
  }  
  
  vector<double> point(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3);
  double distance = depth_plane_[3];
  for (int c = 0; c < 3; c++)
    distance -= depth_plane_[c] * point[c];
  distance = abs(distance);
  if (point_cloud[pixel * 3 + 2] < 0)
    distance = 0;
  if (distance > input_statistics_.pixel_fitting_distance_threshold)
    return false;
      
  vector<double> normal(normals.begin() + pixel * 3, normals.begin() + (pixel + 1) * 3);
  double cos_value = 0;
  for (int c = 0; c < 3; c++)
    cos_value += normal[c] * depth_plane_[c];
  double angle = acos(min(abs(cos_value), 1.0));
  if (sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2)) < 0.000001)
    angle = 0;
  if (angle > input_statistics_.pixel_fitting_angle_threshold)
    return false;
    
  return true;

  // const double DEPTH_FITTING_THRESHOLD = sqrt(segment_statistics_.depth_residual);
    // double u = pixel % IMAGE_WIDTH_ - CAMERA_PARAMETERS_[1];
    // double v = pixel / IMAGE_WIDTH_ - CAMERA_PARAMETERS_[2];
    
    // double fitted_disp = u * disp_plane_[0] + v * disp_plane_[1] + disp_plane_[2];
    // double fitted_depth = fitted_disp != 0 ? 1 / fitted_depth : 0;
    // double input_depth = point_cloud[pixel * 3 + 2];
    // if (abs(fitted_depth  - input_depth) > DEPTH_FITTING_THRESHOLD)
    //   return false;
  // } else {
  //   const double DISP_FITTING_THRESHOLD = sqrt(segment_statistics_.disp_residual);
  //   vector<double> point(3);
  //   point[0] = pixel % IMAGE_WIDTH_ - CAMERA_PARAMETERS_[1];
  //   point[1] = pixel / IMAGE_WIDTH_ - CAMERA_PARAMETERS_[2];
  //   point[2] = 1 / point_cloud[pixel * 3 + 2];

  //   double fitted_disp = point[0] * disp_plane_[0] + point[1] * disp_plane_[1] + disp_plane_[2];
  //   double input_disp = point[2];
  //   if (abs(fitted_disp  - input_disp) > DISP_FITTING_THRESHOLD)
  //     return false;
  // }

  // vector<double> normal(normals.begin() + pixel * 3, normals.begin() + (pixel + 1) * 3);
  // double cos_value = 0;
  // for (int c = 0; c < 3; c++)
  //   cos_value += normal[c] * depth_plane_[c];
  // double angle = acos(abs(cos_value));
  
  // return angle < input_statistics_.fitting_angle_threshold;
}

double Segment::calcFittedDepth(const vector<double> &depth_plane, const int pixel)
{
  vector<double> disp_plane = vector<double>(3);
  disp_plane[0] = depth_plane[0] / (CAMERA_PARAMETERS_[0] * depth_plane[3]);
  disp_plane[1] = depth_plane[1] / (CAMERA_PARAMETERS_[0] * depth_plane[3]);
  disp_plane[2] = depth_plane[2] / depth_plane[3];
  
  double u = pixel % IMAGE_WIDTH_ - CAMERA_PARAMETERS_[1];
  double v = pixel / IMAGE_WIDTH_ - CAMERA_PARAMETERS_[2];
  
  double disp = disp_plane[0] * u + disp_plane[1] * v + disp_plane[2];
  double depth = disp != 0 ? 1 / disp : 0;
  if (depth <= 0)
    depth = 1000000;
  return depth;
}

vector<int> Segment::findLargestConnectedComponent(const vector<double> &point_cloud, const vector<int> &pixels)
{
  vector<int> new_segment_pixels;
  vector<bool> segment_mask(NUM_PIXELS_, false);
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
    segment_mask[*pixel_it] = true;

  vector<bool> visited_pixel_mask(NUM_PIXELS_, false);
  map<int, vector<int> > connected_components;
  int connected_component_index = 0;
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
    if (visited_pixel_mask[*pixel_it] == true)
      continue;

    vector<int> connected_component;
    vector<int> border_pixels;
    border_pixels.push_back(*pixel_it);
    visited_pixel_mask[*pixel_it] = true;
    while (true) {
      vector<int> new_border_pixels;
      for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
	connected_component.push_back(*border_pixel_it);
	//	double depth = point_cloud[*border_pixel_it * 3 + 2];
        vector<int> neighbor_pixels;
	int x = *border_pixel_it % IMAGE_WIDTH_;
	int y = *border_pixel_it / IMAGE_WIDTH_;
	if (x > 0)
	  neighbor_pixels.push_back(*border_pixel_it - 1);
	if (x < IMAGE_WIDTH_ - 1)
	  neighbor_pixels.push_back(*border_pixel_it + 1);
	if (y > 0)
	  neighbor_pixels.push_back(*border_pixel_it - IMAGE_WIDTH_);
	if (y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(*border_pixel_it + IMAGE_WIDTH_);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(*border_pixel_it - 1 - IMAGE_WIDTH_);
	if (x > 0 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(*border_pixel_it - 1 + IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y > 0)
	  neighbor_pixels.push_back(*border_pixel_it + 1 - IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(*border_pixel_it + 1 + IMAGE_WIDTH_);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  if (segment_mask[*neighbor_pixel_it] == true && visited_pixel_mask[*neighbor_pixel_it] == false) {
	    new_border_pixels.push_back(*neighbor_pixel_it);
	    visited_pixel_mask[*neighbor_pixel_it] = true;
	  }
	}
      }
      if (new_border_pixels.size() == 0)
	break;
      border_pixels = new_border_pixels;
    }
    connected_components[connected_component_index] = connected_component;
    connected_component_index++;
  }  

  int max_num_pixels = 0;
  int max_num_pixels_component_index = -1;
  for (map<int, vector<int> >::const_iterator component_it = connected_components.begin(); component_it != connected_components.end(); component_it++) {
    if (component_it->second.size() > max_num_pixels) {
      max_num_pixels_component_index = component_it->first;
      max_num_pixels = component_it->second.size();
    }
  }
  
  return connected_components[max_num_pixels_component_index];
}

double Segment::calcAngle(const vector<double> &normals, const int pixel)
{
  if (segment_type_ == -1)
    return M_PI / 2;
  vector<double> normal(normals.begin() + pixel * 3, normals.begin() + (pixel + 1) * 3);
  if (sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2)) < 0.000001)
    return 0;
  vector<double> surface_normal = segment_type_ == 0 ? vector<double>(depth_plane_.begin(), depth_plane_.begin() + 3) : vector<double>(normals_.begin() + pixel * 3, normals_.begin() + (pixel + 1) * 3);
  double cos_value = 0;
  for (int c = 0; c < 3; c++)
    cos_value += normal[c] * surface_normal[c];
  double angle = acos(min(abs(cos_value), 1.0));
  return angle;
}

bool checkCloseParallelSegments(const Segment &segment_1, const Segment &segment_2, const int pixel, const double distance_threshold, const double angle_threshold)
{
  if (segment_1.segment_type_ != 0 || segment_2.segment_type_ != 0)
    return false;
  vector<double> depth_plane_1 = segment_1.depth_plane_;
  vector<double> depth_plane_2 = segment_2.depth_plane_;
  double cos_value = 0;
  for (int c = 0; c < 3; c++)
    cos_value += depth_plane_1[c] * depth_plane_2[c];
  double angle = acos(min(abs(cos_value), 1.0));
  double depth_1 = segment_1.getDepth(pixel);
  double depth_2 = segment_2.getDepth(pixel);
  if (depth_1 < 0 || depth_2 < 0)
    return false;
  // if (pixel == 75 * 196 + 186)
  //   cout << abs(depth_1 - depth_2) << '\t' << angle << endl;
  return abs(depth_1 - depth_2) < distance_threshold && angle < angle_threshold;
}


void Segment::calcDistanceMap()
{
  vector<double> distances(NUM_PIXELS_, 1000000);
  distance_map_ = vector<int>(NUM_PIXELS_);

  vector<int> border_pixels;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    if (segment_mask_[pixel] == false)
      continue;
    distance_map_[pixel] = pixel;
    distances[pixel] = 0;
    
    vector<int> neighbor_pixels;
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    if (x > 0)
      neighbor_pixels.push_back(pixel - 1);
    if (x < IMAGE_WIDTH_ - 1)
      neighbor_pixels.push_back(pixel + 1);
    if (y > 0)
      neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
    if (y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
    if (x > 0 && y > 0)
      neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
    if (x > 0 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y > 0)
      neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      if (segment_mask_[*neighbor_pixel_it] == false) {
	border_pixels.push_back(pixel);
	break;
      }
    }
  }
  
  while (border_pixels.size() > 0) {
    vector<int> new_border_pixels;
    for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
      int pixel = *border_pixel_it;
      double distance = distances[pixel];
      vector<int> neighbor_pixels;
      int x = pixel % IMAGE_WIDTH_;
      int y = pixel / IMAGE_WIDTH_;
      // if (distance_map_[IMAGE_HEIGHT_ / 2 * IMAGE_WIDTH_ + IMAGE_WIDTH_ / 2] == 0)
      //   cout << x << '\t' << y << '\t' << border_pixel_distance << endl;
      if (x > 0)
	neighbor_pixels.push_back(pixel - 1);
      if (x < IMAGE_WIDTH_ - 1)
	neighbor_pixels.push_back(pixel + 1);
      if (y > 0)
	neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
      if (y < IMAGE_HEIGHT_ - 1)
	neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
      if (x > 0 && y > 0)
	neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
      if (x > 0 && y < IMAGE_HEIGHT_ - 1)
	neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
      if (x < IMAGE_WIDTH_ - 1 && y > 0)
	neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
      if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
	neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);      
      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	int neighbor_pixel = *neighbor_pixel_it;
	double distance_delta = sqrt(pow(*neighbor_pixel_it % IMAGE_WIDTH_ - pixel % IMAGE_WIDTH_, 2) + pow(*neighbor_pixel_it / IMAGE_WIDTH_ - pixel / IMAGE_WIDTH_, 2));
	if (distance + distance_delta < distances[neighbor_pixel]) {
	  distance_map_[neighbor_pixel] = pixel;
	  distances[neighbor_pixel] = distance + distance_delta;
	  new_border_pixels.push_back(neighbor_pixel);
	}
      }
    }
    border_pixels = new_border_pixels;
  }
}

int Segment::calcDistanceOffset(const int pixel_1, const int pixel_2)
{
  if (distance_map_[pixel_1] == pixel_2)
    return 1;
  if (distance_map_[pixel_2] == pixel_1)
    return -1;
  return 0;
}

// void Segment::calcDistanceMap()
// {
//   distance_map_ = vector<int>(NUM_PIXELS_, -1);
  
//   vector<int> border_pixels;
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     if (segment_mask_[pixel] == false)
//       continue;
//     distance_map_[pixel] = 0;
    
//     vector<int> neighbor_pixels;
//     int x = pixel % IMAGE_WIDTH_;
//     int y = pixel / IMAGE_WIDTH_;
//     if (x > 0)
//       neighbor_pixels.push_back(pixel - 1);
//     if (x < IMAGE_WIDTH_ - 1)
//       neighbor_pixels.push_back(pixel + 1);
//     if (y > 0)
//       neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
//     if (y < IMAGE_HEIGHT_ - 1)
//       neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
//     if (x > 0 && y > 0)
//       neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
//     if (x > 0 && y < IMAGE_HEIGHT_ - 1)
//       neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
//     if (x < IMAGE_WIDTH_ - 1 && y > 0)
//       neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
//     if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
//       neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
//     for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
//       if (segment_mask_[*neighbor_pixel_it] == false) {
// 	border_pixels.push_back(pixel);
// 	break;
//       }
//     }
//   }
//   int border_pixel_distance = 0;
//   while (border_pixels.size() > 0) {
//     vector<int> new_border_pixels;
//     for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
//       int pixel = *border_pixel_it;
      
//       vector<int> neighbor_pixels;
//       int x = pixel % IMAGE_WIDTH_;
//       int y = pixel / IMAGE_WIDTH_;
//       // if (distance_map_[IMAGE_HEIGHT_ / 2 * IMAGE_WIDTH_ + IMAGE_WIDTH_ / 2] == 0)
//       //   cout << x << '\t' << y << '\t' << border_pixel_distance << endl;
//       if (x > 0)
//         neighbor_pixels.push_back(pixel - 1);
//       if (x < IMAGE_WIDTH_ - 1)
//         neighbor_pixels.push_back(pixel + 1);
//       if (y > 0)
//         neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
//       if (y < IMAGE_HEIGHT_ - 1)
//         neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
//       if (x > 0 && y > 0)
//         neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
//       if (x > 0 && y < IMAGE_HEIGHT_ - 1)
//         neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
//       if (x < IMAGE_WIDTH_ - 1 && y > 0)
//         neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
//       if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
//         neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);      
//       for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
//         int neighbor_pixel = *neighbor_pixel_it;
//         if (distance_map_[neighbor_pixel] == -1) {
// 	  distance_map_[neighbor_pixel] = border_pixel_distance + 1;
//           new_border_pixels.push_back(neighbor_pixel);
// 	}
//       }
//     }
//     border_pixels = new_border_pixels;
//     border_pixel_distance += 1;
//   }

//   // if (distance_map_[99] == 0) {
//   //   Mat segment_mask_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
//   //   Mat distance_mask_image(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
//   //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//   //     int distance = distance_map_[pixel];
//   //     if (distance == 0)
//   // 	segment_mask_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = 255;
//   //     distance_mask_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = distance * 3;
//   //   }
//   //   imwrite("Test/segment_mask_image.bmp", segment_mask_image);
//   //   imwrite("Test/distance_mask_image.bmp", distance_mask_image);
//   //   exit(1);
//   // }
// }

// int Segment::calcDistanceOffset(const int pixel_1, const int pixel_2)
// {
//   return distance_map_[pixel_1] - distance_map_[pixel_2];
// }


// bool Segment::buildSubSegment(const Mat &image, const vector<double> &point_cloud, const vector<double> &normals, const vector<int> &visible_pixels)
// {
//   if (use_sub_segment_ == true)
//     return false;
//   use_sub_segment_ = true;
  
//   vector<bool> visible_pixel_mask(NUM_PIXELS_, false);
//   int min_x = 1000000, max_x = -1000000, min_y = 1000000, max_y = -1000000;
//   for (vector<int>::const_iterator pixel_it = visible_pixels.begin(); pixel_it != visible_pixels.end(); pixel_it++) {
//     visible_pixel_mask[*pixel_it] = true;
//     int x = *pixel_it % IMAGE_WIDTH_;
//     int y = *pixel_it / IMAGE_WIDTH_;
//     if (x < min_x)
//       min_x = x;
//     if (x > max_x)
//       max_x = x;
//     if (y < min_y)
//       min_y = y;
//     if (y > max_y)
//       max_y = y;
//   }
  
//   int sub_segment_start_x = max(min_x - (max_x - min_x), 0);
//   int sub_segment_end_x = min(max_x + (max_x - min_x), IMAGE_WIDTH_ - 1);
//   int sub_segment_width = sub_segment_end_x - sub_segment_start_x + 1;
//   int sub_segment_start_y = max(min_y - (max_y - min_y), 0);
//   int sub_segment_end_y = min(max_y + (max_y - min_y), IMAGE_WIDTH_ - 1);
//   int sub_segment_height = sub_segment_end_y - sub_segment_start_y + 1;
  
//   vector<bool> pixel_depth_change_mask(NUM_PIXELS_, false);
//   for (int y = sub_segment_start_y; y < sub_segment_start_y + sub_segment_height; y++) {
//     for (int x = sub_segment_start_x; x < sub_segment_start_x + sub_segment_width; x++) {
//       int pixel = y * IMAGE_WIDTH_ + x;
//       double depth = point_cloud[pixel * 3 + 2];
//       vector<double> normal(normals.begin() + pixel * 3, normals.begin() + (pixel + 1) * 3);
//       vector<int> neighbor_pixels;
//       if (x > sub_segment_start_x)
// 	neighbor_pixels.push_back(pixel - 1);
//       if (x < sub_segment_start_x + sub_segment_width - 1)
// 	neighbor_pixels.push_back(pixel + 1);
//       if (y > sub_segment_start_y)
// 	neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
//       if (y < sub_segment_start_y + sub_segment_height - 1)
// 	neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
//       if (x > sub_segment_start_x && y > sub_segment_start_y)
// 	neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
//       if (x > sub_segment_start_x && y < sub_segment_start_y + sub_segment_height - 1)
// 	neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
//       if (x < sub_segment_start_x + sub_segment_width - 1 && y > sub_segment_start_y)
// 	neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
//       if (x < sub_segment_start_x + sub_segment_width - 1 && y < sub_segment_start_y + sub_segment_height - 1)
// 	neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
//       for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
// 	double neighbor_depth = point_cloud[*neighbor_pixel_it * 3 + 2];
// 	vector<double> neighbor_normal(normals.begin() + *neighbor_pixel_it * 3, normals.begin() + (*neighbor_pixel_it + 1) * 3);
// 	if (abs(depth - neighbor_depth) > input_statistics_.depth_change_smoothness_threshold) {
// 	  pixel_depth_change_mask[pixel] = true;
// 	  break;
// 	}
// 	double cos_value = 0;
// 	for (int c = 0; c < 3; c++)
// 	  cos_value += normal[c] * neighbor_normal[c];
// 	double angle = acos(abs(cos_value));
// 	if (angle > input_statistics_.fitting_angle_threshold) {
// 	  pixel_depth_change_mask[pixel] = true;
// 	  break;
// 	}
//       }
//     }
//   }
  
//   vector<vector<int> > connected_components;
//   int connected_component_index = 0;
//   vector<bool> visited_pixel_mask(NUM_PIXELS_, false);
//   for (vector<int>::const_iterator pixel_it = visible_pixels.begin(); pixel_it != visible_pixels.end(); pixel_it++) {
//     if (visited_pixel_mask[*pixel_it] == true || pixel_depth_change_mask[*pixel_it] == true)
//       continue;

//     vector<int> connected_component;
//     vector<int> border_pixels;
//     border_pixels.push_back(*pixel_it);
//     visited_pixel_mask[*pixel_it] = true;
//     while (true) {
//       vector<int> new_border_pixels;
//       for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
// 	connected_component.push_back(*border_pixel_it);
// 	int pixel = *border_pixel_it;
// 	vector<int> neighbor_pixels;
// 	int x = pixel % IMAGE_WIDTH_;
// 	int y = pixel / IMAGE_WIDTH_;
// 	if (x > sub_segment_start_x)
// 	  neighbor_pixels.push_back(pixel - 1);
// 	if (x < sub_segment_start_x + sub_segment_width - 1)
// 	  neighbor_pixels.push_back(pixel + 1);
// 	if (y > sub_segment_start_y)
// 	  neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
// 	if (y < sub_segment_start_y + sub_segment_height - 1)
// 	  neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
// 	if (x > sub_segment_start_x && y > sub_segment_start_y)
// 	  neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
// 	if (x > sub_segment_start_x && y < sub_segment_start_y + sub_segment_height - 1)
// 	  neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
// 	if (x < sub_segment_start_x + sub_segment_width - 1 && y > sub_segment_start_y)
// 	  neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
// 	if (x < sub_segment_start_x + sub_segment_width - 1 && y < sub_segment_start_y + sub_segment_height - 1)
// 	  neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
// 	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
// 	  if (visible_pixel_mask[*neighbor_pixel_it] == true && visited_pixel_mask[*neighbor_pixel_it] == false && pixel_depth_change_mask[*neighbor_pixel_it] == false) {
// 	    new_border_pixels.push_back(*neighbor_pixel_it);
// 	    visited_pixel_mask[*neighbor_pixel_it] = true;
// 	  }
// 	}
//       }
//       if (new_border_pixels.size() == 0)
// 	break;
//       border_pixels = new_border_pixels;
//     }
//     connected_components.push_back(connected_component);
//     connected_component_index++;
//   }

//   segment_pixels_.clear();
//   for (vector<vector<int> >::const_iterator connected_component_it = connected_components.begin(); connected_component_it != connected_components.end(); connected_component_it++)
//     if (connected_component_it->size() > segment_pixels_.size())
//       segment_pixels_ = *connected_component_it;
//   if (segment_pixels_.size() < 3) {
//     segment_pixels_.clear();
//     return false;
//   }


//   vector<int> distance_map(NUM_PIXELS_, -1);
//   for (vector<int>::const_iterator pixel_it = segment_pixels_.begin(); pixel_it != segment_pixels_.end(); pixel_it++)
//     distance_map[*pixel_it] = 0;
//   int border_pixel_distance = 0;
//   vector<int> border_pixels = segment_pixels_;
//   while (border_pixels.size() > 0) {
//     vector<int> new_border_pixels;
//     for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
//       int pixel = *border_pixel_it;
//       vector<int> neighbor_pixels;
//       int x = pixel % IMAGE_WIDTH_;
//       int y = pixel / IMAGE_WIDTH_;
//       if (x > sub_segment_start_x)
// 	neighbor_pixels.push_back(pixel - 1);
//       if (x < sub_segment_start_x + sub_segment_width- 1)
// 	neighbor_pixels.push_back(pixel + 1);
//       if (y > sub_segment_start_y)
// 	neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
//       if (y < sub_segment_start_y + sub_segment_height - 1)
// 	neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
//       if (x > sub_segment_start_x && y > sub_segment_start_y)
// 	neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
//       if (x > sub_segment_start_x && y < sub_segment_start_y + sub_segment_height - 1)
// 	neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
//       if (x < sub_segment_start_x + sub_segment_width - 1 && y > sub_segment_start_y)
// 	neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
//       if (x < sub_segment_start_x + sub_segment_width - 1 && y < sub_segment_start_y + sub_segment_height - 1)
// 	neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
//       for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
// 	if (distance_map[*neighbor_pixel_it] != -1) {
// 	  distance_map_[*neighbor_pixel_it] = border_pixel_distance + 1;
// 	  new_border_pixels.push_back(*neighbor_pixel_it);
// 	}
//       }
//     }
//     border_pixels = new_border_pixels;
//     border_pixel_distance += 1;
//   }

  
//   border_pixels = segment_pixels_;
//   visited_pixel_mask.assign(NUM_PIXELS_, false);
//   while (border_pixels.size() > 0) {
//     vector<int> new_border_pixels;
//     for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
//       int pixel = *border_pixel_it;
//       vector<int> neighbor_pixels;
//       int x = pixel % IMAGE_WIDTH_;
//       int y = pixel / IMAGE_WIDTH_;
//       if (x > sub_segment_start_x)
// 	neighbor_pixels.push_back(pixel - 1);
//       if (x < sub_segment_start_x + sub_segment_width - 1)
// 	neighbor_pixels.push_back(pixel + 1);
//       if (y > sub_segment_start_y)
// 	neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
//       if (y < sub_segment_start_y + sub_segment_height - 1)
// 	neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
//       if (x > sub_segment_start_x && y > sub_segment_start_y)
// 	neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
//       if (x > sub_segment_start_x && y < sub_segment_start_y + sub_segment_height - 1)
// 	neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
//       if (x < sub_segment_start_x + sub_segment_width - 1 && y > sub_segment_start_y)
// 	neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
//       if (x < sub_segment_start_x + sub_segment_width - 1 && y < sub_segment_start_y + sub_segment_height - 1)
// 	neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
//       for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
// 	if (visited_pixel_mask[*neighbor_pixel_it] == false && distance_map[*neighbor_pixel_it] > distance_map[pixel] && pixel_depth_change_mask[*neighbor_pixel_it] == false) {
// 	  segment_pixels_.push_back(*neighbor_pixel_it);
// 	  new_border_pixels.push_back(*neighbor_pixel_it);
// 	  visited_pixel_mask[*neighbor_pixel_it] = true;
// 	}
//       }
//     }
//     border_pixels = new_border_pixels;
//     border_pixel_distance += 1;
//   }

//   for (vector<int>::const_iterator pixel_it = segment_pixels_.begin(); pixel_it != segment_pixels_.end(); pixel_it++)
//     depth_map_[*pixel_it] = point_cloud[*pixel_it * 3 + 2];
  
//   calcColorStatistics(image, segment_pixels_);
//   calcSegmentMaskInfo();
//   calcConfidence();

//   return true;
// }


void Segment::fitBSplineSurface(const Mat &image, const vector<double> &point_cloud, const std::vector<double> &normals, const vector<int> &pixels)
{
  // BSpline bspline(point_cloud, visible_pixels, 1, IMAGE_WIDTH_, IMAGE_HEIGHT_);
  // vector<double> depth_map = bspline.getDepthMap();
  // double plane_fitting_error = calcFittingError(point_cloud, visible_pixels);

  segment_pixels_ = findLargestConnectedComponent(point_cloud, pixels);
  if (segment_pixels_.size() < 3) {
    fitParallelSurface(point_cloud, normals, segment_pixels_);
    return;
  }
  
  //segment_type_ = 2;
  BSplineSurface surface(point_cloud, segment_pixels_, IMAGE_WIDTH_, IMAGE_HEIGHT_, 5, 5, segment_type_);
  depth_map_ = surface.getDepthMap();
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   if (depth_map_[pixel] < 0 || depth_map_[pixel] > 10)
  //     cout << pixel << '\t' << depth_map_[pixel] << endl;
  normals_ = calcNormals(calcPointCloud(), IMAGE_WIDTH_, IMAGE_HEIGHT_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    double scale = 0;
    for (int c = 0; c < 3; c++)
      scale += pow(normals_[pixel * 3 + c], 2);
    // if (abs(scale - 1) > 0.000001)
    //   cout << pixel << '\t' << scale << endl;
  }
}

void Segment::refitSegmentKeepingGeometry(const Mat &image, const vector<double> &point_cloud, const std::vector<double> &normals, const vector<int> &pixels)
{
  segment_pixels_ = pixels;
  if (segment_pixels_.size() >= 3)
    calcColorStatistics(image, segment_pixels_);
  calcSegmentMaskInfo();
  //calcConfidence();
}

// double Segment::calcFittingError(const vector<double> &point_cloud, const vector<int> &visible_pixels)
// {
// }

double Segment::calcDistance2D(const double x_ratio, const double y_ratio) const
{
  double x = IMAGE_WIDTH_ * x_ratio;
  double y = IMAGE_HEIGHT_ * y_ratio;
  int lower_x = max(floor(x), 0.0);
  int upper_x = min(static_cast<int>(ceil(x)), IMAGE_WIDTH_ - 1);
  int lower_y = max(floor(y), 0.0);
  int upper_y = min(static_cast<int>(ceil(y)), IMAGE_HEIGHT_ - 1);
  if (lower_x == upper_x && lower_y == upper_y)
    return distance_map_[lower_y * IMAGE_WIDTH_ + lower_x];
  else if (lower_x == upper_x)
    return distance_map_[lower_y * IMAGE_WIDTH_ + lower_x] * (upper_y - y) + distance_map_[upper_y * IMAGE_WIDTH_ + lower_x] * (y - lower_y);
  else if (lower_y == upper_y)
    return distance_map_[lower_y * IMAGE_WIDTH_ + lower_x] * (upper_x - x) + distance_map_[lower_y * IMAGE_WIDTH_ + upper_x] * (x - lower_x);
  else {
    double area_1 = (x - lower_x) * (y - lower_y);
    double area_2 = (x - lower_x) * (upper_y - y);
    double area_3 = (upper_x - x) * (y - lower_y);
    double area_4 = (upper_x - x) * (upper_y - y);
    double distance_1 = distance_map_[lower_y * IMAGE_WIDTH_ + lower_x];
    double distance_2 = distance_map_[upper_y * IMAGE_WIDTH_ + lower_x];
    double distance_3 = distance_map_[lower_y * IMAGE_WIDTH_ + upper_x];
    double distance_4 = distance_map_[upper_y * IMAGE_WIDTH_ + upper_x];
    
    return distance_1 * area_4 + distance_2 * area_3 + distance_3 * area_2 + distance_4 * area_1;
  }
}

// bool Segment::mergeSegment(const Segment &other_segment, const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals)
// {
//   vector<double> other_depth_plane = other_segment.getDepthPlane();
//   double cos_value = 0;
//   for (int c = 0; c < 3; c++)
//     cos_value += other_depth_plane[c] * depth_plane_[c];
//   double angle = acos(min(abs(cos_value), 1.0));
//   if (angle > input_statistics_.fitting_angle_threshold)
//     return false;
  
//   vector<int> other_segment_pixels = other_segment.getSegmentPixels();
//   vector<double> other_depth_map = other_segment.getDepthMap();
//   const int NUM_CONFLICT_PIXELS_THRESHOLD = other_segment_pixels.size() * 0.1;
//   int num_conflict_pixels = 0;
//   for (vector<int>::const_iterator pixel_it = other_segment_pixels.begin(); pixel_it != other_segment_pixels.end(); pixel_it++)
//     if (abs(other_depth_map[*pixel_it] - depth_map_[*pixel_it]) > input_statistics_.fitting_distance_threshold)
//       num_conflict_pixels++;
//   if (num_conflict_pixels > NUM_CONFLICT_PIXELS_THRESHOLD)
//     return false;
  
//   vector<bool> merged_segment_mask(NUM_PIXELS_, false);
//   int min_x = IMAGE_WIDTH_;
//   int max_x = -1;
//   int min_y = IMAGE_HEIGHT_;
//   int max_y = -1;
//   for (vector<int>::const_iterator pixel_it = segment_pixels_.begin(); pixel_it != segment_pixels_.end(); pixel_it++) {
//     merged_segment_mask[*pixel_it] = true;
//     int x = *pixel_it % IMAGE_WIDTH_;
//     int y = *pixel_it / IMAGE_WIDTH_;
//     if (x < min_x)
//       min_x = x;
//     if (x > max_x)
//       max_x = x;
//     if (y < min_y)
//       min_y = y;
//     if (y > max_y)
//       max_y = y;
//   }
//   int other_min_x = IMAGE_WIDTH_;
//   int other_max_x = -1;
//   int other_min_y = IMAGE_HEIGHT_;
//   int other_max_y = -1;
//   for (vector<int>::const_iterator pixel_it = other_segment_pixels.begin(); pixel_it != other_segment_pixels.end(); pixel_it++) {
//     merged_segment_mask[*pixel_it] = true;
//     int x = *pixel_it % IMAGE_WIDTH_;
//     int y = *pixel_it / IMAGE_WIDTH_;
//     if (x < other_min_x)
//       other_min_x = x;
//     if (x > other_max_x)
//       other_max_x = x;
//     if (y < other_min_y)
//       other_min_y = y;
//     if (y > other_max_y)
//       other_max_y = y;
//   }

//   if (other_min_x > max_x || other_max_x < min_x || other_min_y > max_y || other_max_y < min_y)
//     return false;

//   vector<int> merged_segment_pixels;
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
//     if (merged_segment_mask[pixel])
//       merged_segment_pixels.push_back(pixel);

//   fitDepthPlane(image, point_cloud, normals, merged_segment_pixels);
//   calcColorStatistics(image, segment_pixels_);
//   calcSegmentMaskInfo();
//   calcConfidence();
//   return true;
// }

vector<int> Segment::projectToOtherViewpoints(const int pixel, const double viewpoint_movement)
{
  vector<int> projected_pixels;
  int x = pixel % IMAGE_WIDTH_;
  int y = pixel / IMAGE_WIDTH_;
  // double u = x - CAMERA_PARAMETERS_[1];
  // double v = y - CAMERA_PARAMETERS_[2];
  // //double depth = 1 / ((plane(0) * u + plane(1) * v + plane(2)) / plane(3));
    
  // double disp = disp_plane_[0] * u + disp_plane_[1] * v + disp_plane_[2];
  //double depth = disp != 0 ? 1 / disp : 0;

  double depth = depth_map_[pixel];
  if (depth <= 0)
    return projected_pixels;
  int delta = round(viewpoint_movement / depth * CAMERA_PARAMETERS_[0]);
  if (x - delta >= 0)
    projected_pixels.push_back(pixel - delta);
  if (x + delta < IMAGE_WIDTH_)
    projected_pixels.push_back(pixel + delta + NUM_PIXELS_);
  if (y - delta >= 0)
    projected_pixels.push_back(pixel - delta * IMAGE_WIDTH_ + NUM_PIXELS_ * 2);
  if (y + delta < IMAGE_HEIGHT_)
    projected_pixels.push_back(pixel + delta * IMAGE_WIDTH_ + NUM_PIXELS_ * 3);
  return projected_pixels;
}

vector<double> Segment::calcPointCloud()
{
  vector<double> point_cloud(NUM_PIXELS_ * 3);
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    double depth = depth_map_[pixel];
    point_cloud[pixel * 3 + 0] = (pixel % IMAGE_WIDTH_ - CAMERA_PARAMETERS_[1]) / CAMERA_PARAMETERS_[0] * depth;
    point_cloud[pixel * 3 + 1] = (pixel / IMAGE_WIDTH_ - CAMERA_PARAMETERS_[2]) / CAMERA_PARAMETERS_[0] * depth;
    point_cloud[pixel * 3 + 2] = depth;
  }
  return point_cloud;
}

int Segment::getSegmentType() const
{
  return segment_type_;
}

double Segment::calcColorDiff(const Vec3f &color_1, const Vec3f &color_2)
{
  double color_diff = sqrt(pow(color_1[1] * cos(color_1[0] * M_PI / 180) - color_2[1] * cos(color_2[0] * M_PI / 180), 2) + pow(color_1[1] * sin(color_1[0] / 180 * M_PI) - color_2[1] * sin(color_2[0] / 180 * M_PI), 2)); // + pow(color_1[2] * 0.1 - color_2[2] * 0.1, 2));
  return color_diff;
}

Segment upsampleSegment(const Segment &segment, const cv::Mat &new_image, const std::vector<double> &new_point_cloud, const std::vector<double> &new_normals, const vector<double> &new_camera_parameters, const vector<int> &pixels)
{
  Segment new_segment(new_image.cols, new_image.rows, new_camera_parameters, segment.penalties_, segment.input_statistics_);
  if (segment.segment_type_ == 0) {
    new_segment.segment_type_ = segment.segment_type_;
    new_segment.depth_plane_ = segment.depth_plane_;
    new_segment.calcDepthMap();
    new_segment.segment_pixels_ = pixels;
    new_segment.calcColorStatistics(new_image, new_segment.segment_pixels_);
    new_segment.calcSegmentMaskInfo();
  } else if (segment.segment_type_ > 0)
    new_segment.fitBSplineSurface(new_image, new_point_cloud, new_normals, pixels);
  else
    new_segment.fitParallelSurface(new_point_cloud, new_normals, pixels);
  return new_segment;
}

vector<int> Segment::deleteInvalidPixels(const vector<double> &point_cloud, const vector<int> &pixels)
{
  vector<int> new_pixels;
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
    if (point_cloud[*pixel_it * 3 + 2] > 0)
      new_pixels.push_back(*pixel_it);
  return new_pixels;
}

double Segment::calcDistance(const vector<double> &point_cloud, const int pixel)
{
  if (point_cloud[pixel * 3 + 2] < 0)
    return input_statistics_.pixel_fitting_distance_threshold;
    
  if (segment_type_ == 0) {
    double distance = depth_plane_[3];
    for (int c = 0; c < 3; c++)
      distance -= depth_plane_[c] * point_cloud[pixel * 3 + c];
    return abs(distance);
  } else {
    double depth = depth_map_[pixel];
    if (depth <= 0)
      return input_statistics_.pixel_fitting_distance_threshold;
    vector<double> point(3);
    point[0] = (pixel % IMAGE_WIDTH_ - CAMERA_PARAMETERS_[1]) / CAMERA_PARAMETERS_[0] * depth;
    point[1] = (pixel / IMAGE_WIDTH_ - CAMERA_PARAMETERS_[2]) / CAMERA_PARAMETERS_[0] * depth;
    point[2] = depth;
    double distance = 0;
    for (int c = 0; c < 3; c++)
      distance += pow((point_cloud[pixel * 3 + c] - point[c]) * normals_[pixel * 3 + c], 2);
    return sqrt(distance);
  }
}

// double Segment::calcDistance(const vector<double> &point)
// {
//   if (segment_type_ == 0) {
//     double distance = depth_plane_[3];
//     for (int c = 0; c < 3; c++)
//       distance -= depth_plane_[c] * point[c];
//     return abs(distance);
//   } else {
//     return 1;
//   }
// }
