#ifndef SEGMENTATION_PROPOSAL_GENERATOR_H__
#define SEGMENTATION_PROPOSAL_GENERATOR_H__

#include <vector>
#include <opencv2/core/core.hpp>

#include "../base/ProposalGenerator.h"

class SegmentationProposalGenerator : public ProposalGenerator
{
 public:
 SegmentationProposalGenerator(const cv::Mat &image) : image_(image.clone()), IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows) {};
  LabelSpace getProposal() const;
  std::vector<int> getInitialSolution() const;

 private:
  cv::Mat image_;
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;
};

#endif
