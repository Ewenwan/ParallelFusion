#ifndef OPTICAL_FLOW_PROPOSAL_GENERATOR_H__
#define OPTICAL_FLOW_PROPOSAL_GENERATOR_H__

#include <vector>
#include <opencv2/core/core.hpp>

#include "../base/ProposalGenerator.h"

class OpticalFlowProposalGenerator : public ProposalGenerator<std::pair<double, double> >
{
 public:
 OpticalFlowProposalGenerator(const cv::Mat &image_1, const cv::Mat &image_2, const int NUM_PROPOSAL_SOLUTIONS_THRESHOLD, const std::vector<std::pair<double, double> > &ground_truth_solution) : image_1_(image_1.clone()), image_2_(image_2.clone()), IMAGE_WIDTH_(image_1.cols), IMAGE_HEIGHT_(image_1.rows), NUM_PROPOSAL_SOLUTIONS_THRESHOLD_(NUM_PROPOSAL_SOLUTIONS_THRESHOLD), ground_truth_flows_(ground_truth_solution) {};
  LabelSpace<std::pair<double, double> > getProposal();
  std::vector<std::pair<double, double> > getInitialSolution() const;
  void writeSolution(const std::vector<std::pair<double, double> > &solution, const int iteration, const int thread_index) const;
  //void setGroundTruthSolution(const std::vector<std::pair<double, double> > &ground_truth_solution) { ground_truth_flows_ = ground_truth_solution; };
  
 private:
  cv::Mat image_1_;
  cv::Mat image_2_;
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;

  const int NUM_PROPOSAL_SOLUTIONS_THRESHOLD_;

  int num_proposed_solutions_;
  std::string proposal_name_;

  std::vector<std::pair<double, double> > ground_truth_flows_;

  
  LabelSpace<std::pair<double, double> > generateProposalPyrLK();
  LabelSpace<std::pair<double, double> > generateProposalFarneback();
  LabelSpace<std::pair<double, double> > generateProposalLayerWise();
  LabelSpace<std::pair<double, double> > generateProposalMoveAround();
  LabelSpace<std::pair<double, double> > generateProposalShift();
  LabelSpace<std::pair<double, double> > generateProposalCluster();
};

#endif
