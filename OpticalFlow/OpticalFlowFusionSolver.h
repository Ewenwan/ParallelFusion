#ifndef OPTICAL_FLOW_FUSION_SOLVER_H__
#define OPTICAL_FLOW_FUSION_SOLVER_H__

#include <vector>
#include <opencv2/core/core.hpp>

#include "../base/FusionSolver.h"

class OpticalFlowFusionSolver : public FusionSolver<std::pair<double, double> >
{
 public:
 OpticalFlowFusionSolver(const cv::Mat &image_1, const cv::Mat &image_2) : image_1_(image_1.clone()), image_2_(image_2.clone()), IMAGE_WIDTH_(image_1.cols), IMAGE_HEIGHT_(image_1.rows) {};
  std::vector<std::pair<double, double> > solve(const LabelSpace<std::pair<double, double> > &label_space, double &energy) const;

  double checkSolutionEnergy(const std::vector<std::pair<double, double> > &solution);
  
 private:
  cv::Mat image_1_;
  cv::Mat image_2_;
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;

  const double SMOOTHNESS_TERM_WEIGHT_ = 1;
  

  double calcDataCost(const int pixel, const std::pair<double, double> &flow) const;
  double calcSmoothnessCost(const int pixel_1, const int pixel_2, const std::pair<double, double> &flow_1, const std::pair<double, double> &flow_2) const;
};
  
#endif
