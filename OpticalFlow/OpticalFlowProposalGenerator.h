#ifndef OPTICAL_FLOW_PROPOSAL_GENERATOR_H__
#define OPTICAL_FLOW_PROPOSAL_GENERATOR_H__

#include <vector>
#include <opencv2/core/core.hpp>

#include "../base/ProposalGenerator.h"

namespace flow_fusion {
  class OpticalFlowProposalGenerator : public ParallelFusion::ProposalGenerator<ParallelFusion::LabelSpace<std::pair<double, double> > > {

    typedef ParallelFusion::LabelSpace<std::pair<double, double> > LABELSPACE;
    public:
    OpticalFlowProposalGenerator(const cv::Mat &image_1, const cv::Mat &image_2) : image_1_(
											    image_1.clone()), image_2_(image_2.clone()), IMAGE_WIDTH_(image_1.cols), IMAGE_HEIGHT_(image_1.rows) {};
	
    void getProposals(LABELSPACE& proposal_label_space, const LABELSPACE& current_solution, const int N);
        
      std::vector<std::pair<double, double> > getInitialSolution() const;

      void writeSolution(const std::pair<double, LABELSPACE> &solution, const int thread_index, const int iteration) const;
        
      //void setGroundTruthSolution(const std::vector<std::pair<double, double> > &ground_truth_solution) { ground_truth_flows_ = ground_truth_solution; };

      void getAllProposals(std::vector<LABELSPACE> &proposals);
      

    private:
        cv::Mat image_1_;
        cv::Mat image_2_;
        const int IMAGE_WIDTH_;
        const int IMAGE_HEIGHT_;

        int num_proposed_solutions_;
        std::string proposal_name_;

	
	std::vector<std::pair<double, double> > current_solution_;


	  void setCurrentSolution(const LABELSPACE &current_solution_label_space);
        
        void generateProposalPyrLK(LABELSPACE& proposal_label_space);

        void generateProposalFarneback(LABELSPACE& proposal_label_space);

        void generateProposalLayerWise(LABELSPACE& proposal_label_space);

        void generateProposalNearestNeighbor(LABELSPACE& proposal_label_space);

        void generateProposalMoveAround(LABELSPACE& proposal_label_space);

        void generateProposalShift(LABELSPACE& proposal_label_space);

        void generateProposalCluster(LABELSPACE& proposal_label_space);

        void generateProposalDisturb(LABELSPACE& proposal_label_space);
    };
}

#endif
