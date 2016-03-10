#include "OpticalFlowProposalGenerator.h"

#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

#include "../base/cv_utils/cv_utils.h"
#include "OpticalFlowCalculation.h"
#include "OpticalFlowUtils.h"

using namespace std;
using namespace cv;
using namespace ParallelFusion;

namespace flow_fusion {
    LabelSpace <pair<double, double>> OpticalFlowProposalGenerator::getProposal() {
      // const int NUM_LABELS = 5;
      // vector<int> node_labels(NUM_LABELS);
      // for (int c = 0; c < NUM_LABELS; c++)
      //   node_labels[c] = c;

      // return LabelSpace(vector<vector<int> >(IMAGE_WIDTH_ * IMAGE_HEIGHT_, node_labels));

      num_proposed_solutions_ = 0;

      LabelSpace <pair<double, double>> proposal_label_space(current_solution_);
      //int index = 0;
      while (num_proposed_solutions_ < NUM_PROPOSAL_SOLUTIONS_THRESHOLD_) {
        int proposal_type = rand() % 4 + 4 * (cv_utils::randomProbability() < 0.3);
        //int proposal_type = 5;
        //if (index == 1)
        //proposal_type = 5;
        //index++;
        switch (proposal_type) {
          case 0:
            proposal_label_space += generateProposalPyrLK();
                break;
          case 1:
            proposal_label_space += generateProposalFarneback();
                break;
          case 2:
            proposal_label_space += generateProposalLayerWise();
                break;
          case 3:
            proposal_label_space += generateProposalNearestNeighbor();
                break;
          case 4:
            proposal_label_space += generateProposalMoveAround();
                break;
          case 5:
            proposal_label_space += generateProposalShift();
                break;
          case 6:
            proposal_label_space += generateProposalCluster();
                break;
          case 7:
            proposal_label_space += generateProposalDisturb();
                break;
        }
      }
      return proposal_label_space;
    }

    vector<pair<double, double> > OpticalFlowProposalGenerator::getInitialSolution() const {
      return vector<pair<double, double> >(IMAGE_WIDTH_ * IMAGE_HEIGHT_);
    }

    void OpticalFlowProposalGenerator::writeSolution(const std::vector<pair<double, double> > &solution,
                                                     const int iteration, const int thread_index) const {
      Mat solution_image = drawFlows(solution, IMAGE_WIDTH_, IMAGE_HEIGHT_);
      imwrite("Test/solution_image_" + to_string(iteration) + "_" + to_string(thread_index) + ".png", solution_image);
      cout << "error: " << calcFlowsDiff(solution, ground_truth_flows_, IMAGE_WIDTH_, IMAGE_HEIGHT_) << endl;
      writeFlows(solution, IMAGE_WIDTH_, IMAGE_HEIGHT_,
                 "Cache/flows_" + to_string(iteration) + "_" + to_string(thread_index) + ".flo");
    }

    LabelSpace <pair<double, double>> OpticalFlowProposalGenerator::generateProposalPyrLK() {
      cout << "generate proposal PyrLK" << endl;
      proposal_name_ = "PryLK";

      const int NUM_LEVELS = rand() % 5;
      vector<pair<double, double> > proposal_flows = calcFlowsPyrLK(image_1_, image_2_, NUM_LEVELS);

      num_proposed_solutions_++;

      imwrite("Test/proposal_flows_PyrLK.png", drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
      return LabelSpace<pair<double, double> >(proposal_flows);
    }

    LabelSpace <pair<double, double>> OpticalFlowProposalGenerator::generateProposalFarneback() {
      cout << "generate proposal Farneback" << endl;
      proposal_name_ = "Farneback";

      const int NUM_LEVELS = rand() % 5;
      const int POLY_N = (rand() % 4) * 2 + 1;
      const int FLAGS = rand() % 2 == 0 ? 0 : OPTFLOW_FARNEBACK_GAUSSIAN;
      vector<pair<double, double> > proposal_flows = calcFlowsFarneback(image_1_, image_2_, NUM_LEVELS, POLY_N, FLAGS);

      num_proposed_solutions_++;

      imwrite("Test/proposal_flows_Farneback.png", drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
      return LabelSpace<pair<double, double> >(proposal_flows);
    }

    LabelSpace <pair<double, double>> OpticalFlowProposalGenerator::generateProposalLayerWise() {
      cout << "generate proposal LayerWise" << endl;
      proposal_name_ = "LayerWise";

      const double ALPHA = rand() % 2 == 0 ? 0.01 : (rand() % 2 == 0 ? 0.005 : 0.002);
      const double RATIO = rand() % 2 == 0 ? 0.5 : 0.75;
      vector<pair<double, double> > proposal_flows = calcFlowsLayerWise(image_1_, image_2_, ALPHA, RATIO);

      num_proposed_solutions_++;

      imwrite("Test/proposal_flows_LayerWise.png", drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
      return LabelSpace<pair<double, double> >(proposal_flows);
    }

    LabelSpace <pair<double, double>> OpticalFlowProposalGenerator::generateProposalNearestNeighbor() {
      cout << "generate proposal NearestNeighbor" << endl;
      proposal_name_ = "NearestNeighbor";

      const int WINDOW_SIZE = 5 + 2 * (rand() % 6);
      vector<pair<double, double> > proposal_flows = calcFlowsNearestNeighbor(image_1_, image_2_, WINDOW_SIZE);

      num_proposed_solutions_++;
      imwrite("Test/proposal_flows_NearestNeighbor.png", drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
      return LabelSpace<pair<double, double> >(proposal_flows);
    }

    LabelSpace <pair<double, double>> OpticalFlowProposalGenerator::generateProposalMoveAround() {
      cout << "generate proposal MoveAround" << endl;
      proposal_name_ = "MoveAround";

      const double FLOW_MAX_RANGE = 7;
      const int NUM_PROPOSAL_SOLUTIONS = min(4, NUM_PROPOSAL_SOLUTIONS_THRESHOLD_ - num_proposed_solutions_);
      num_proposed_solutions_ += NUM_PROPOSAL_SOLUTIONS;

      const double RADIUS = cv_utils::randomProbability() * FLOW_MAX_RANGE;
      const double CENTER_X = cv_utils::drawFromNormalDistribution() / 2 * FLOW_MAX_RANGE;
      const double CENTER_Y = cv_utils::drawFromNormalDistribution() / 2 * FLOW_MAX_RANGE;

      LabelSpace <pair<double, double>> proposed_label_space;
      for (int proposal_solution_index = 0;
           proposal_solution_index < NUM_PROPOSAL_SOLUTIONS; proposal_solution_index++) {
        double shift_x = CENTER_X + (cv_utils::randomProbability() * 2 - 1) * RADIUS;
        double shift_y = CENTER_Y + (cv_utils::randomProbability() * 2 - 1) * RADIUS;
        vector<pair<double, double> > proposal_flows = shiftFlows(current_solution_, IMAGE_WIDTH_, IMAGE_HEIGHT_,
                                                                  shift_x, shift_y);
        proposed_label_space += LabelSpace<pair<double, double> >(proposal_flows);

        imwrite("Test/proposal_flows_MoveAround_" + to_string(proposal_solution_index) + ".png",
                drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
      }
      return proposed_label_space;
    }

    LabelSpace <pair<double, double>> OpticalFlowProposalGenerator::generateProposalShift() {
      cout << "generate proposal Shift" << endl;
      proposal_name_ = "Shift";

      const double FLOW_MAX_RANGE = 7;
      const int NUM_PROPOSAL_SOLUTIONS = min(4, NUM_PROPOSAL_SOLUTIONS_THRESHOLD_ - num_proposed_solutions_);
      num_proposed_solutions_ += NUM_PROPOSAL_SOLUTIONS;

      const double RADIUS = 3;
      const pair<double, double> SHIFT_DIRECTION = rand() % 2 == 0 ? make_pair(0, 1) : make_pair(1, 0);

      LabelSpace <pair<double, double>> proposed_label_space;
      for (int proposal_solution_index = 0;
           proposal_solution_index < NUM_PROPOSAL_SOLUTIONS; proposal_solution_index++) {
        double shift_x = (proposal_solution_index + 1) * SHIFT_DIRECTION.first * RADIUS / NUM_PROPOSAL_SOLUTIONS;
        double shift_y = (proposal_solution_index + 1) * SHIFT_DIRECTION.second * RADIUS / NUM_PROPOSAL_SOLUTIONS;
        vector<pair<double, double> > proposal_flows = shiftFlows(current_solution_, IMAGE_WIDTH_, IMAGE_HEIGHT_,
                                                                  shift_x, shift_y);
        proposed_label_space += LabelSpace<pair<double, double> >(proposal_flows);

        imwrite("Test/proposal_flows_Shift_" + to_string(proposal_solution_index) + ".png",
                drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
      }
      return proposed_label_space;
    }

    LabelSpace <pair<double, double>> OpticalFlowProposalGenerator::generateProposalDisturb() {
      cout << "generate proposal Disturb" << endl;
      proposal_name_ = "Disturb";

      const double DISTURB_STDDEV = 1;
      vector<pair<double, double> > proposal_flows = disturbFlows(current_solution_, IMAGE_WIDTH_, IMAGE_HEIGHT_,
                                                                  DISTURB_STDDEV);
      imwrite("Test/proposal_flows_disturb.png", drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
      num_proposed_solutions_++;
      return LabelSpace<pair<double, double> >(proposal_flows);;
    }

    LabelSpace <pair<double, double>> OpticalFlowProposalGenerator::generateProposalCluster() {
      cout << "generate proposal Cluster" << endl;
      proposal_name_ = "Cluster";

      const int NUM_CLUSTERS = 64;
      const int IMAGE_BORDER = 10;
      const double RADIUS = 3;
      const int NUM_PROPOSAL_SOLUTIONS = min(4, NUM_PROPOSAL_SOLUTIONS_THRESHOLD_ - num_proposed_solutions_);
      num_proposed_solutions_ += NUM_PROPOSAL_SOLUTIONS;

      Mat flow_points((IMAGE_WIDTH_ - IMAGE_BORDER * 2) * (IMAGE_HEIGHT_ - IMAGE_BORDER * 2), 1, CV_32FC2);
      for (int y = IMAGE_BORDER; y < IMAGE_HEIGHT_ - IMAGE_BORDER; y++)
        for (int x = IMAGE_BORDER; x < IMAGE_WIDTH_ - IMAGE_BORDER; x++)
          flow_points.at<Vec2f>((y - IMAGE_BORDER) * (IMAGE_WIDTH_ - IMAGE_BORDER * 2) + (x - IMAGE_BORDER)) = Vec2f(
                  current_solution_[y * IMAGE_WIDTH_ + x].first, current_solution_[y * IMAGE_WIDTH_ + x].second);

      Mat labels, centers;
      kmeans(flow_points, NUM_CLUSTERS, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3,
             KMEANS_PP_CENTERS, centers);
      // for (int i = 0; i < NUM_CLUSTERS; i++)
      //   cout << "center: " << centers.at<Vec2f>(i) << endl;

      LabelSpace <pair<double, double>> proposal_label_space;
      for (int proposal_solution_index = 0;
           proposal_solution_index < NUM_PROPOSAL_SOLUTIONS; proposal_solution_index++) {
        double shift_x = proposal_solution_index == 0 ? 0 : cv_utils::drawFromNormalDistribution() * RADIUS;
        double shift_y = proposal_solution_index == 0 ? 0 : cv_utils::drawFromNormalDistribution() * RADIUS;

        vector<pair<double, double> > proposal_flows = current_solution_;
        for (int y = IMAGE_BORDER; y < IMAGE_HEIGHT_ - IMAGE_BORDER; y++) {
          for (int x = IMAGE_BORDER; x < IMAGE_WIDTH_ - IMAGE_BORDER; x++) {
            int label = labels.at<int>((y - IMAGE_BORDER) * (IMAGE_WIDTH_ - IMAGE_BORDER * 2) + (x - IMAGE_BORDER));
            Vec2f center = centers.at<Vec2f>(label);
            pair<double, double> flow(center[0] + shift_x, center[1] + shift_y);
            proposal_flows[y * IMAGE_WIDTH_ + x] = flow;
          }
        }
        proposal_label_space += LabelSpace<pair<double, double> >(proposal_flows);

        imwrite("Test/proposal_flows_Cluster_" + to_string(proposal_solution_index) + ".png",
                drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
      }
      return proposal_label_space;
    }
}