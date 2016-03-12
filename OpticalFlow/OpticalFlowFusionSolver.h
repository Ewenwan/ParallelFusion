#ifndef OPTICAL_FLOW_FUSION_SOLVER_H__
#define OPTICAL_FLOW_FUSION_SOLVER_H__

#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../base/FusionSolver.h"

namespace flow_fusion {
  typedef ParallelFusion::LabelSpace<std::pair<double, double> > LABELSPACE;
  
  class OpticalFlowFusionSolver : public ParallelFusion::FusionSolver<LABELSPACE> {
  public:

    
        OpticalFlowFusionSolver(const cv::Mat &image_1, const cv::Mat &image_2);

	void solve(const LABELSPACE &proposals, const ParallelFusion::SolutionType<LABELSPACE>& current_solution, ParallelFusion::SolutionType<LABELSPACE>& solution);

	//given a solution, evaluate energy
	double evaluateEnergy(const LABELSPACE & solution) const;
        

    private:
        cv::Mat image_1_;
        cv::Mat image_2_;
        cv::Mat image_1_high_freq_;
        cv::Mat image_2_high_freq_;
        cv::Mat image_1_gray_;
        cv::Mat image_2_gray_;

        const int IMAGE_WIDTH_;
        const int IMAGE_HEIGHT_;

        const double SMOOTHNESS_TERM_WEIGHT_ = 1;

        std::vector<std::map<int, double> > pixel_neighbor_weights_;


        double calcDataCost(const int pixel, const std::pair<double, double> &flow) const;

        double calcSmoothnessCost(const int pixel_1, const int pixel_2, const std::pair<double, double> &flow_1,
                                  const std::pair<double, double> &flow_2) const;

        void calcNeighborInfo();

        std::vector<double> readColorVec(const bool left_or_right, const int x, const int y) const;

        std::vector<double> getImageColor(const bool left_or_right, const double x, const double y) const;
    };
}
#endif
