//
// Created by yanhang on 2/28/16.
//

#ifndef DYNAMICSTEREO_PROPOSAL_H
#define DYNAMICSTEREO_PROPOSAL_H

#include "../stereo_base/depth.h"
#include "../stereo_base/file_io.h"
#include <glog/logging.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
namespace sce_stereo {

// interface for proposal creator
class Proposal {
public:
  virtual void genProposal(std::vector<stereo_base::Depth> &proposals) = 0;
};

class ProposalSegPln : public Proposal {
public:
  // constructor input:
  //  images_: reference image
  //  noisyDisp_: disparity map from only unary term
  //  num_proposal: number of proposal to generate. NOTE: currently fixed to 7
  ProposalSegPln(const stereo_base::FileIO &file_io_, const cv::Mat &image_,
                 const stereo_base::Depth &noisyDisp_,
                 const int dispResolution_, const std::string &method_,
                 const int num_proposal_ = 7);
  virtual void genProposal(std::vector<stereo_base::Depth> &proposals);

protected:
  void fitDisparityToPlane(const std::vector<std::vector<int>> &seg,
                           stereo_base::Depth &planarDisp, int id);

  // input:
  //  pid: id of parameter setting
  //  seg: stores the segmentation result. seg[i] stores pixel indices of region
  //  i
  virtual void segment(const int pid, std::vector<std::vector<int>> &seg) = 0;

  const stereo_base::FileIO &file_io;
  const stereo_base::Depth &noisyDisp;
  const cv::Mat &image;
  const int num_proposal;
  std::vector<double> params;
  std::vector<double> mults;

  const int w;
  const int h;

  const int dispResolution;

  const std::string method;
};

class ProposalSegPlnMeanshift : public ProposalSegPln {
public:
  ProposalSegPlnMeanshift(const stereo_base::FileIO &file_io_,
                          const cv::Mat &image_,
                          const stereo_base::Depth &noisyDisp_,
                          const int dispResolution_,
                          const int num_proposal_ = 10);

protected:
  virtual void segment(const int pid, std::vector<std::vector<int>> &seg);
};

class ProposalSegPlnGbSegment : public ProposalSegPln {
public:
  ProposalSegPlnGbSegment(const stereo_base::FileIO &file_io_,
                          const cv::Mat &image_,
                          const stereo_base::Depth &noisyDisp_,
                          const int dispResolution_,
                          const int num_proposal_ = 10);

protected:
  virtual void segment(const int pid, std::vector<std::vector<int>> &seg);
};

} // namespace dynamic_stereo

#endif // DYNAMICSTEREO_PROPOSAL_H
