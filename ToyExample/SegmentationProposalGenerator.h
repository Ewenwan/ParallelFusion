#ifndef SEGMENTATION_PROPOSAL_GENERATOR_H__
#define SEGMENTATION_PROPOSAL_GENERATOR_H__

#include <vector>
#include <opencv2/core/core.hpp>

#include "../base/ProposalGenerator.h"

class SegmentationProposalGenerator : public ProposalGenerator<int>
{
 public:
 SegmentationProposalGenerator(const cv::Mat &image) : image_(image.clone()), IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows) {};
  LabelSpace<int> getProposal();
  std::vector<int> getInitialSolution() const;
  void writeSolution(const std::vector<int> &solution, const int iteration, const int thread_index) const;
  
 private:
  cv::Mat image_;
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;
};

#endif
