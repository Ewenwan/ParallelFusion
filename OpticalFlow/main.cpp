//#include <gflags/gflags.h>

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <memory>

#include "../base/BaseDataStructures.h"
#include "../base/FusionThread.h"
#include "../base/ParallelFusionPipeline.h"
//#include "../base/cv_utils/cv_utils.h"
#include "OpticalFlowCalculation.h"
#include "OpticalFlowUtils.h"
#include "OpticalFlowFusionSolver.h"
#include "OpticalFlowProposalGenerator.h"

// DEFINE_int32(num_fusion_iterations, 1, "The number of iterations in fusion pipeline.");
// DEFINE_int32(num_threads, 1, "The number of threads.");

using namespace std;
using namespace cv;


int main()
{
  //  PipelineParams pipeline_params(FLAGS_num_threads, FLAGS_num_fusion_iterations);

  Mat image_1 = imread("other-data/Dimetrodon/frame10.png");
  Mat image_2 = imread("other-data/Dimetrodon/frame11.png");

  Mat blurred_image_1, blurred_image_2;
  GaussianBlur(image_1, blurred_image_1, cv::Size(11, 11), 0, 0);
  GaussianBlur(image_2, blurred_image_2, cv::Size(11, 11), 0, 0);
  subtract(image_1, blurred_image_1, image_1);
  subtract(image_2, blurred_image_2, image_2);

  vector<pair<double, double> > ground_truth_flows = readFlowsFromFile("other-gt-flow/Dimetrodon/flow10.flo");

  //cout << OPTFLOW_FARNEBACK_GAUSSIAN << endl;
  //vector<pair<double, double> > flows = calcFlowsPyrLK(image_1, image_2, 1);
  vector<pair<double, double> > flows = calcFlowsFarneback(image_1, image_2, 3, 1, 0);
  OpticalFlowFusionSolver fusion_solver(image_1, image_2);
  cout << fusion_solver.checkSolutionEnergy(flows) << endl;
  exit(1);
  //vector<pair<double, double> > flows = calcFlowsLayerWise(image_1, image_2, 0.01, 0.75);
  //vector<pair<double, double> > flows = calcFlowsBrox(image_1, image_2, 0.01, 0.75);
  
  // for (int pixel = 0; pixel < image_1.cols * image_1.rows; pixel++)
  //   for (int c = 0; c < 2; c++)
  //     cout << flows[pixel][c] << '\t' << ground_truth_flows[pixel][c] << endl;
  //imwrite("Test/flow_image.png", drawFlows(flows, image_1.cols, image_1.rows, 5));
  // for (int c = 0; c < 100; c++)
  //   cout << cv_utils::drawFromNormalDistribution(0, 0.1) << endl;
  // exit(1);

  PipelineParams pipeline_params(1, 5, 2);
  vector<double> fusion_thread_master_likelihood_vec(pipeline_params.NUM_THREADS, 0);
  //fusion_thread_master_likelihood_vec[pipeline_params.NUM_THREADS - 1] = 1;
  vector<unique_ptr<FusionThread<pair<double, double> > > > fusion_threads;
  vector<shared_ptr<ProposalGenerator<pair<double, double> > > > proposal_generators;
  for (int fusion_thread_index = 0; fusion_thread_index < pipeline_params.NUM_THREADS; fusion_thread_index++)
    proposal_generators.push_back(dynamic_pointer_cast<ProposalGenerator<pair<double, double> > >(shared_ptr<OpticalFlowProposalGenerator>(new OpticalFlowProposalGenerator(image_1, image_2, pipeline_params.NUM_PROPOSAL_SOLUTIONS_THRESHOLD, ground_truth_flows))));
  vector<shared_ptr<FusionSolver<pair<double, double> > > > fusion_solvers;
  for (int fusion_thread_index = 0; fusion_thread_index < pipeline_params.NUM_THREADS; fusion_thread_index++)
    fusion_solvers.push_back(dynamic_pointer_cast<FusionSolver<pair<double, double> > >(shared_ptr<OpticalFlowFusionSolver>(new OpticalFlowFusionSolver(image_1, image_2))));
  for (int fusion_thread_index = 0; fusion_thread_index < pipeline_params.NUM_THREADS; fusion_thread_index++)			  
    fusion_threads.push_back(move(unique_ptr<FusionThread<pair<double, double> > >(new FusionThread<pair<double, double> >(proposal_generators[fusion_thread_index], fusion_solvers[fusion_thread_index], fusion_thread_master_likelihood_vec[fusion_thread_index]))));
    
  vector<pair<double, double> > fused_solution = parallelFuse(fusion_threads, pipeline_params, vector<pair<double, double> >(image_1.cols * image_1.rows));

  cout << calcFlowsDiff(fused_solution, ground_truth_flows, image_1.cols, image_1.rows) << endl;;
  return 0;
}
