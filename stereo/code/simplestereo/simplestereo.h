//
// Created by yanhang on 3/7/16.
//

#ifndef PARALLELFUSION_SIMPLESTEREO_H
#define PARALLELFUSION_SIMPLESTEREO_H

#include <Eigen/Eigen>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "../stereo_base/depth.h"
#include "../stereo_base/file_io.h"
#include "optimization.h"

namespace simple_stereo {
class SimpleStereo {
public:
  SimpleStereo(const stereo_base::FileIO &file_io_, const int anchor_,
               const int dispResolution_, const int downsample_,
               const double weight_smooth_, const int num_threads_,
               const int num_proposal_, const int exchange_interval_,
               const int exchange_amount_);

  ~SimpleStereo() { model->clear(); }

  void initMRF();

  void computeMatchingCost();

  void assignSmoothWeight();

  void runStereo();

  inline int getWidth() const { return width; }

  inline int getHeight() const { return height; }

private:
  typedef int EnergyType;

  const stereo_base::FileIO &file_io;
  const int anchor;
  const int downsample;
  const int num_threads;
  const int num_proposals, exchange_interval, exchange_amount;

  std::vector<cv::Mat> images;
  MRFModel<int> *model;
  stereo_base::Depth unaryDisp;
  int width;
  int height;
};

} // namespace simple_stereo

#endif // PARALLELFUSION_SIMPLESTEREO_H
