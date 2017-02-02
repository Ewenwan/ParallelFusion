#ifndef SEGMENTATION_FUSION_SOLVER_H__
#define SEGMENTATION_FUSION_SOLVER_H__

#include <opencv2/core/core.hpp>
#include <vector>

#include "../base/FusionSolver.h"

class SegmentationFusionSolver : public ParallelFusion::FusionSolver<int> {
public:
  SegmentationFusionSolver(const cv::Mat &image)
      : image_(image.clone()), IMAGE_WIDTH_(image.cols),
        IMAGE_HEIGHT_(image.rows){};
  std::vector<int> solve(const ParallelFusion::LabelSpace<int> &label_space,
                         double &energy) const;

private:
  cv::Mat image_;
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;
};

#endif
