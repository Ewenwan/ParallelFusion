#ifndef __LayerDepthMap__Segment__
#define __LayerDepthMap__Segment__

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "DataStructure.h"
//#include "BSpline.h"


class Segment{

 public:
  Segment(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<double> &camera_parameters, const std::vector<int> &pixels, const RepresenterPenalties &penalties, const DataStatistics &input_statistics = DataStatistics(), const int segment_type = 0);
  Segment(const int image_width, const int image_height, const std::vector<double> &camera_parameters, const RepresenterPenalties &penalties, const DataStatistics &statistics);
  //Segment(const Segment &segment);
  Segment();
  
  //~Segment();

  friend std::ostream & operator <<(std::ostream &out_str, const Segment &segment);
  friend std::istream & operator >>(std::istream &in_str, Segment &segment);

  friend bool checkCloseParallelSegments(const Segment &segment_1, const Segment &segment_2, const int pixel, const double distance_threshold, const double angle_threshold);
  
  Segment &operator = (const Segment &segment);
  
  double calcFittingScore(const int pixel, const double depth);
  void refitSegmentKeepingGeometry(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &visible_pixels);
  void refitSegment(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels);
  
  std::vector<Segment> calcDividedSegments(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<double> &camera_parameters, const std::vector<int> &visible_pixels, const RepresenterPenalties &penalties, const DataStatistics &input_statistics, const int num_divided_segments, const double residual_decrease_threshold) const;

  //std::vector<int> getFittedPixels(const std::vector<double> &point_cloud, const std::vector<int> &pixels);
  
  double predictColorLikelihood(const int pixel, const cv::Vec3f hsv_color) const;

  void setGMM(const cv::Ptr<cv::ml::EM> GMM);
  void setGMM(const cv::FileNode GMM_file_node);
  
  cv::Ptr<cv::ml::EM> getGMM() const;

  std::vector<double> getDepthMap() const;
  double getDepth(const int pixel) const;
  double getDepth(const double x_ratio, const double y_ratio) const;
  std::vector<double> getDepthPlane() const;
  double getConfidence() const;
  DataStatistics getSegmentStatistics() const;
  int getNumSegmentPixels() const;
  std::vector<int> getSegmentPixels() const;
  int getType() const;
    
  //double calcDistanceRatio2D(const int pixel);
  double calcDistance2D(const double ratio_x, const double ratio_y) const;
  bool checkPixelFitting(const cv::Mat &hsv_image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const int pixel) const;
  double calcAngle(const std::vector<double> &normals, const int pixel);
  int calcDistanceOffset(const int pixel_1, const int pixel_2);

  //bool buildSubSegment(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &visible_pixels);

  bool mergeSegment(const Segment &other_segment, const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals);

  std::vector<int> projectToOtherViewpoints(const int pixel, const double viewpoint_movement);

  int getSegmentType() const;

  double calcDistance(const std::vector<double> &point_cloud, const int pixel);
  

  friend Segment upsampleSegment(const Segment &segment, const cv::Mat &new_image, const std::vector<double> &new_point_cloud, const std::vector<double> &new_normals, const std::vector<double> &new_camera_parameters, const std::vector<int> &pixels);

  
 private:
  int IMAGE_WIDTH_;
  int IMAGE_HEIGHT_;

  int NUM_PIXELS_;
  std::vector<double> CAMERA_PARAMETERS_;
  
  
  RepresenterPenalties penalties_;
  DataStatistics input_statistics_;
  DataStatistics segment_statistics_;

  std::vector<int> segment_pixels_;
  //  std::vector<int> visible_pixels_;
  std::vector<double> disp_plane_;
  std::vector<double> depth_plane_;
  std::vector<double> depth_map_;
  std::vector<double> normals_;
  
  int segment_type_;

  cv::Ptr<cv::ml::EM> GMM_;
  
  double segment_confidence_;

  std::vector<bool> segment_mask_;
  double segment_radius_;
  double segment_center_x_;
  double segment_center_y_;

  //bool DEPTH_OR_DISP_FITTING_;
  
  std::vector<int> distance_map_;


  void fitDispPlane(const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels, const std::vector<int> &distance_to_boundaries = std::vector<int>());
  void fitDispPlaneRansac(const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels);
  void fitDispPlaneRobustly(const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels);
  void fitDepthPlaneRansac(const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels);
  void fitDepthPlane(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels);
  void fitBSplineSurface(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels);
  void fitParallelSurface(const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels);
  void calcDepthMap(const std::vector<double> &point_cloud = std::vector<double>(), const std::vector<int> &fitted_pixels = std::vector<int>());
  void calcColorStatistics(const cv::Mat &image, const std::vector<int> &pixels);
  void calcConfidence();
  void calcSegmentMaskInfo();
  double calcFittedDepth(const std::vector<double> &depth_plane, const int pixel);
  void calcDistanceMap();
  
  std::vector<int> findLargestConnectedComponent(const std::vector<double> &point_cloud, const std::vector<int> &pixels);
  
  void writeSegmentImage(const std::string filename);

  double calcFittingError(const std::vector<double> &point_cloud, const std::vector<int> &visible_pixels);
  std::vector<double> calcPointCloud();

  double calcColorDiff(const cv::Vec3f &color_1, const cv::Vec3f &color_2);

  std::vector<int> deleteInvalidPixels(const std::vector<double> &point_cloud, const std::vector<int> &pixels);
};


std::map<int, Segment> calcSegments(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<int> &segmentation, const RepresenterPenalties &penalties, const DataStatistics &statistics);

/* double calcDispSvar(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<int> &segmentation, const RepresenterPenalties &penalties, const InputStatistics &statistics); */
/* double calcDepthSvar(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<int> &segmentation, const RepresenterPenalties &penalties, const InputStatistics &statistics); */

#endif /* defined(__LayerDepthMap__Segment__) */
