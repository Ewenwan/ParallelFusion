#ifndef OPTICAL_FLOW_PROPOSAL_GENERATOR_H__
#define OPTICAL_FLOW_PROPOSAL_GENERATOR_H__

#include <vector>
#include <map>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "../base/ProposalGenerator.h"

namespace flow_fusion {
  struct Solution
  {
  Solution() : thread_id(-1), time(-1), energy(-1), error(-1) {};
  Solution(const int _thread_id, const int _time, const double _energy, const double _error, const std::vector<int> &_selected_threads, std::vector<std::pair<double, double> > _solution_labels) : thread_id(_thread_id), time(_time), energy(_energy), error(_error), selected_threads(_selected_threads), solution_labels(_solution_labels) {};

    int thread_id;
    int time;
    double energy;
    double error;
    std::vector<int> selected_threads;
    std::vector<std::pair<double, double> > solution_labels;

    friend std::ostream &operator <<(std::ostream &out_str, Solution &solution)
    {
      out_str << solution.thread_id << '\t' << solution.time << '\t' << solution.energy << '\t' << solution.error << std::endl;
      out_str << solution.selected_threads.size() << std::endl;
      for (std::vector<int>::const_iterator thread_it = solution.selected_threads.begin(); thread_it != solution.selected_threads.end(); thread_it++)
	out_str << *thread_it << '\t';
      out_str << std::endl;
      return out_str;
    }
    friend std::istream &operator >>(std::istream &in_str, Solution &solution)
    {
      in_str >> solution.thread_id >> solution.time >> solution.energy >> solution.error;
      int num_selected_threads = 0;
      in_str >> num_selected_threads;
      solution.selected_threads.assign(num_selected_threads, 0);
      for (int i = 0; i < num_selected_threads; i++)
        in_str >> solution.selected_threads[i];
      return in_str;
    }
  };
  
  class OpticalFlowProposalGenerator : public ParallelFusion::ProposalGenerator<ParallelFusion::LabelSpace<std::pair<double, double> > > {

    typedef ParallelFusion::LabelSpace<std::pair<double, double> > LABELSPACE;
    public:
  OpticalFlowProposalGenerator(const cv::Mat &image_1, const cv::Mat &image_2, const std::vector<std::pair<double, double> > &ground_truth_flows) : image_1_(image_1.clone()), image_2_(image_2.clone()), IMAGE_WIDTH_(image_1.cols), IMAGE_HEIGHT_(image_1.rows), ground_truth_flows_(ground_truth_flows) {};
	
    void getProposals(LABELSPACE& proposal_label_space, const LABELSPACE& current_solution, const int N);
        
    std::vector<std::pair<double, double> > getInitialSolution() const;

    void writeSolution(const std::pair<double, LABELSPACE> &solution, const int thread_index, const int iteration, const std::vector<int> &selected_threads);
        
      //void setGroundTruthSolution(const std::vector<std::pair<double, double> > &ground_truth_solution) { ground_truth_flows_ = ground_truth_solution; };

      void getAllProposals(std::vector<LABELSPACE> &proposals);
      std::vector<Solution> getSolutions() { return solutions_; };

    private:
        cv::Mat image_1_;
        cv::Mat image_2_;
        const int IMAGE_WIDTH_;
        const int IMAGE_HEIGHT_;

	std::vector<std::pair<double, double> > ground_truth_flows_;
	
        int num_proposed_solutions_;
        std::string proposal_name_;

	
	std::vector<std::pair<double, double> > current_solution_;

	std::vector<Solution> solutions_;


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
