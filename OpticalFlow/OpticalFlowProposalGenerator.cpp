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

  void OpticalFlowProposalGenerator::setCurrentSolution(const LABELSPACE& current_solution_label_space)
  {
    const int NUM_PIXELS = IMAGE_WIDTH_ * IMAGE_HEIGHT_;
    current_solution_.assign(NUM_PIXELS, make_pair(0.0, 0.0));
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
      current_solution_[pixel] = current_solution_label_space.getLabelOfNode(pixel)[0];
  }
  
  void OpticalFlowProposalGenerator::getProposals(LABELSPACE& proposal_label_space, const LABELSPACE& current_solution_label_space, const int N)
  {
    setCurrentSolution(current_solution_label_space);
  
    // if (iteration != 2) {
    //   cout << current_solution_num_surfaces_ << '\t' << current_solution_segments_.size() << endl;
    //   exit(1);
    // }
    // bool test = false;
    // if (test) {
    //   //generateSegmentRefittingProposal();
    //   generateSingleSurfaceExpansionProposal(8);
    //   //generateSingleProposal();
    //   proposal_labels = proposal_labels_;
    //   proposal_num_surfaces = proposal_num_surfaces_;
    //   proposal_segments = proposal_segments_;
    //   proposal_type = proposal_type_;
    //   return true;
    // }
  
    const int NUM_PROPOSAL_TYPES = 6;

    for (int label_space_index = 0; label_space_index < N; label_space_index++) {

      int proposal_type = rand() % NUM_PROPOSAL_TYPES;
      //int proposal_type = 5;
      //if (index == 1)
      //proposal_type = 5;
      //index++;
      LABELSPACE label_space;
      switch (proposal_type) {
      case 0:
        generateProposalPyrLK(label_space);
        break;
      case 1:
        generateProposalFarneback(label_space);
        break;
      case 2:
        generateProposalMoveAround(label_space);
        break;
      case 3:
        generateProposalShift(label_space);
        break;
      case 4:
        generateProposalCluster(label_space);
        break;
      case 5:
        generateProposalDisturb(label_space);
        break;
	// case 7:
	// 	generateProposalLayerWise(label_space);
	// 	break;
	// case 2:
	//   generateProposalNearestNeighbor(label_space);
	//   break;
      }
      
      proposal_label_space.appendSpace(label_space);
    }
    proposal_label_space.appendSpace(current_solution_label_space);
    
    return;
  }
  
  vector<pair<double, double> > OpticalFlowProposalGenerator::getInitialSolution() const {
    return vector<pair<double, double> >(IMAGE_WIDTH_ * IMAGE_HEIGHT_);
  }

  void OpticalFlowProposalGenerator::writeSolution(const std::pair<double, LABELSPACE> &solution, const int thread_index, const int iteration) const
  {
    const int NUM_PIXELS = IMAGE_WIDTH_ * IMAGE_HEIGHT_;
    vector<pair<double, double> > solution_labels(NUM_PIXELS);
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
      solution_labels[pixel] = solution.second.getLabelOfNode(pixel)[0];

    Mat solution_image = drawFlows(solution_labels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    imwrite("Test/solution_image_" + to_string(iteration) + "_" + to_string(thread_index) + ".png", solution_image);
    //      cout << "error: " << calcFlowsDiff(solution, ground_truth_flows_, IMAGE_WIDTH_, IMAGE_HEIGHT_) << endl;
    //  writeFlows(solution, IMAGE_WIDTH_, IMAGE_HEIGHT_,
    //             "Cache/flows_" + to_string(iteration) + "_" + to_string(thread_index) + ".flo");

    time_t timer;
    time(&timer);
    struct tm today = {0};
    today.tm_hour = today.tm_min = today.tm_sec = 0;
    today.tm_mon = 2;
    today.tm_mday = 9;
    today.tm_year = 116;
    LOG(INFO) << difftime(timer, mktime(&today)) << '\t' << iteration << '\t' << thread_index << '\t' << solution.first << endl;
  }

  void OpticalFlowProposalGenerator::generateProposalPyrLK(LABELSPACE& proposal_label_space) {
    cout << "generate proposal PyrLK" << endl;
    proposal_name_ = "PryLK";

    const int NUM_LEVELS = rand() % 5;
    vector<pair<double, double> > proposal_flows = calcFlowsPyrLK(image_1_, image_2_, NUM_LEVELS);

    num_proposed_solutions_++;

    imwrite("Test/proposal_flows_PyrLK.png", drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
    proposal_label_space.setSingleLabels(proposal_flows);
  }

  void OpticalFlowProposalGenerator::generateProposalFarneback(LABELSPACE& proposal_label_space) {
    cout << "generate proposal Farneback" << endl;
    proposal_name_ = "Farneback";

    const int NUM_LEVELS = rand() % 5;
    const int POLY_N = (rand() % 4) * 2 + 1;
    const int FLAGS = rand() % 2 == 0 ? 0 : OPTFLOW_FARNEBACK_GAUSSIAN;
    vector<pair<double, double> > proposal_flows = calcFlowsFarneback(image_1_, image_2_, NUM_LEVELS, POLY_N, FLAGS);

    num_proposed_solutions_++;

    imwrite("Test/proposal_flows_Farneback.png", drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
    proposal_label_space.setSingleLabels(proposal_flows);
  }

  void OpticalFlowProposalGenerator::generateProposalLayerWise(LABELSPACE& proposal_label_space) {
    cout << "generate proposal LayerWise" << endl;
    proposal_name_ = "LayerWise";

    const double ALPHA = rand() % 2 == 0 ? 0.01 : (rand() % 2 == 0 ? 0.005 : 0.002);
    const double RATIO = rand() % 2 == 0 ? 0.5 : 0.75;
    vector<pair<double, double> > proposal_flows = calcFlowsLayerWise(image_1_, image_2_, ALPHA, RATIO);

    num_proposed_solutions_++;

    imwrite("Test/proposal_flows_LayerWise.png", drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
    proposal_label_space.setSingleLabels(proposal_flows);
  }

  void OpticalFlowProposalGenerator::generateProposalNearestNeighbor(LABELSPACE& proposal_label_space) {
    cout << "generate proposal NearestNeighbor" << endl;
    proposal_name_ = "NearestNeighbor";

    const int WINDOW_SIZE = 5 + 2 * (rand() % 6);
    vector<pair<double, double> > proposal_flows = calcFlowsNearestNeighbor(image_1_, image_2_, WINDOW_SIZE);

    num_proposed_solutions_++;
    imwrite("Test/proposal_flows_NearestNeighbor.png", drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
    proposal_label_space.setSingleLabels(proposal_flows);
  }

  void OpticalFlowProposalGenerator::generateProposalMoveAround(LABELSPACE& proposal_label_space) {
    cout << "generate proposal MoveAround" << endl;
    proposal_name_ = "MoveAround";

    const double FLOW_MAX_RANGE = 7;
    const int NUM_PROPOSAL_SOLUTIONS = 1; //min(4, NUM_PROPOSAL_SOLUTIONS_THRESHOLD_ - num_proposed_solutions_);
    num_proposed_solutions_ += NUM_PROPOSAL_SOLUTIONS;

    const double RADIUS = cv_utils::randomProbability() * FLOW_MAX_RANGE;
    const double CENTER_X = cv_utils::drawFromNormalDistribution() / 2 * FLOW_MAX_RANGE;
    const double CENTER_Y = cv_utils::drawFromNormalDistribution() / 2 * FLOW_MAX_RANGE;

    for (int proposal_solution_index = 0;
	 proposal_solution_index < NUM_PROPOSAL_SOLUTIONS; proposal_solution_index++) {
      double shift_x = CENTER_X + (cv_utils::randomProbability() * 2 - 1) * RADIUS;
      double shift_y = CENTER_Y + (cv_utils::randomProbability() * 2 - 1) * RADIUS;
      vector<pair<double, double> > proposal_flows = shiftFlows(current_solution_, IMAGE_WIDTH_, IMAGE_HEIGHT_,
								shift_x, shift_y);
      proposal_label_space.appendSpace(LabelSpace<pair<double, double> >(proposal_flows));

      imwrite("Test/proposal_flows_MoveAround_" + to_string(proposal_solution_index) + ".png",
	      drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
    }
  }

  void OpticalFlowProposalGenerator::generateProposalShift(LABELSPACE& proposal_label_space) {
    cout << "generate proposal Shift" << endl;
    proposal_name_ = "Shift";

    const double FLOW_MAX_RANGE = 7;
    const int NUM_PROPOSAL_SOLUTIONS = 1; //min(4, NUM_PROPOSAL_SOLUTIONS_THRESHOLD_ - num_proposed_solutions_);
    num_proposed_solutions_ += NUM_PROPOSAL_SOLUTIONS;

    const double RADIUS = 3;
    const pair<double, double> SHIFT_DIRECTION = rand() % 2 == 0 ? make_pair(0, 1) : make_pair(1, 0);

    for (int proposal_solution_index = 0;
	 proposal_solution_index < NUM_PROPOSAL_SOLUTIONS; proposal_solution_index++) {
      double shift_x = (proposal_solution_index + 1) * SHIFT_DIRECTION.first * RADIUS / NUM_PROPOSAL_SOLUTIONS;
      double shift_y = (proposal_solution_index + 1) * SHIFT_DIRECTION.second * RADIUS / NUM_PROPOSAL_SOLUTIONS;
      vector<pair<double, double> > proposal_flows = shiftFlows(current_solution_, IMAGE_WIDTH_, IMAGE_HEIGHT_,
								shift_x, shift_y);
      proposal_label_space.appendSpace(LabelSpace<pair<double, double> >(proposal_flows));

      imwrite("Test/proposal_flows_Shift_" + to_string(proposal_solution_index) + ".png",
	      drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
    }
  }

  void OpticalFlowProposalGenerator::generateProposalDisturb(LABELSPACE& proposal_label_space) {
    cout << "generate proposal Disturb" << endl;
    proposal_name_ = "Disturb";

    const double DISTURB_STDDEV = 1;
    vector<pair<double, double> > proposal_flows = disturbFlows(current_solution_, IMAGE_WIDTH_, IMAGE_HEIGHT_,
								DISTURB_STDDEV);
    imwrite("Test/proposal_flows_disturb.png", drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
    num_proposed_solutions_++;
    proposal_label_space.setSingleLabels(proposal_flows);
  }

  void OpticalFlowProposalGenerator::generateProposalCluster(LABELSPACE& proposal_label_space) {
    cout << "generate proposal Cluster" << endl;
    proposal_name_ = "Cluster";

    const int NUM_CLUSTERS = 64;
    const int IMAGE_BORDER = 10;
    const double RADIUS = 3;
    const int NUM_PROPOSAL_SOLUTIONS = 1; //min(4, NUM_PROPOSAL_SOLUTIONS_THRESHOLD_ - num_proposed_solutions_);
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

    vector<pair<double, double> > proposal_flows = current_solution_;
    for (int proposal_solution_index = 0;
	 proposal_solution_index < NUM_PROPOSAL_SOLUTIONS; proposal_solution_index++) {
      double shift_x = proposal_solution_index == 0 ? 0 : cv_utils::drawFromNormalDistribution() * RADIUS;
      double shift_y = proposal_solution_index == 0 ? 0 : cv_utils::drawFromNormalDistribution() * RADIUS;

      for (int y = IMAGE_BORDER; y < IMAGE_HEIGHT_ - IMAGE_BORDER; y++) {
	for (int x = IMAGE_BORDER; x < IMAGE_WIDTH_ - IMAGE_BORDER; x++) {
	  int label = labels.at<int>((y - IMAGE_BORDER) * (IMAGE_WIDTH_ - IMAGE_BORDER * 2) + (x - IMAGE_BORDER));
	  Vec2f center = centers.at<Vec2f>(label);
	  pair<double, double> flow(center[0] + shift_x, center[1] + shift_y);
	  proposal_flows[y * IMAGE_WIDTH_ + x] = flow;
	}
      }
      proposal_label_space.appendSpace(LabelSpace<pair<double, double> >(proposal_flows));

      imwrite("Test/proposal_flows_Cluster_" + to_string(proposal_solution_index) + ".png",
	      drawFlows(proposal_flows, IMAGE_WIDTH_, IMAGE_HEIGHT_));
    }
  }

  void OpticalFlowProposalGenerator::getAllProposals(vector<LabelSpace<pair<double, double> > > &proposals)
  {
    vector<vector<pair<double, double> > > initial_flows;
    for (int num_levels = 1; num_levels <= 5; num_levels += 1) {
      vector<pair<double, double> > proposal_flows = calcFlowsPyrLK(image_1_, image_2_, num_levels);
      initial_flows.push_back(proposal_flows);
    }

    for (int num_levels = 1; num_levels <= 5; num_levels += 1) {
      for (int poly_n = 1; poly_n <= 7; poly_n += 2) {
	for (int flag_index = 0; flag_index <= 0; flag_index++) {
	  const int FLAGS = flag_index == 0 ? 0 : OPTFLOW_FARNEBACK_GAUSSIAN;
	  vector<pair<double, double> > proposal_flows = calcFlowsFarneback(image_1_, image_2_, num_levels, poly_n, FLAGS);
	  initial_flows.push_back(proposal_flows);
	}
      }
    }

    for (vector<vector<pair<double, double> > >::const_iterator initial_flow_it = initial_flows.begin(); initial_flow_it != initial_flows.end(); initial_flow_it++) {
      
      proposals.push_back(LABELSPACE(*initial_flow_it));
      
      current_solution_ = *initial_flow_it;
      
      for (int proposal_index = 0; proposal_index < 4; proposal_index++) {
	LABELSPACE proposal_label_space;
	generateProposalMoveAround(proposal_label_space);
	proposals.push_back(proposal_label_space);
      }
      for (int proposal_index = 0; proposal_index < 4; proposal_index++) {
	LABELSPACE proposal_label_space;
        generateProposalShift(proposal_label_space);
        proposals.push_back(proposal_label_space);
      }
      for (int proposal_index = 0; proposal_index < 1; proposal_index++) {
	LABELSPACE proposal_label_space;
        generateProposalCluster(proposal_label_space);
        proposals.push_back(proposal_label_space);
      }
      for (int proposal_index = 0; proposal_index < 2; proposal_index++) {
        LABELSPACE proposal_label_space;
        generateProposalDisturb(proposal_label_space);
        proposals.push_back(proposal_label_space);
      }
    }
    random_shuffle(proposals.begin(), proposals.end());
  }
}
