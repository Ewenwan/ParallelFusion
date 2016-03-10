//#include <gflags/gflags.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <memory>
#include <ctime>

#include "../base/ParallelFusionPipeline.h"
//#include "../base/cv_utils/cv_utils.h"
#include "OpticalFlowCalculation.h"
#include "OpticalFlowUtils.h"
#include "OpticalFlowFusionSolver.h"
#include "OpticalFlowProposalGenerator.h"

// DEFINE_int32(num_fusion_iterations, 1, "The number of iterations in fusion pipeline.");
// DEFINE_int32(num_threads, 1, "The number of threads.");

//DEFINE_string(dataset_name, "other-data/Dimetrodon", "The dataset image name.");
DEFINE_string(dataset_name, "eval-data/Backyard", "The dataset image name.");
DEFINE_int32(num_proposed_solutions, 4, "The number of proposed solution for each thread.");
DEFINE_bool(write_log, true, "Write log file or not.");
DEFINE_bool(evaluation, false, "Write log file or not.");
//DEFINE_string(dataset_category, "other-data", "The dataset image category.");

DEFINE_int32(num_threads, 2, "The number of threads.");
DEFINE_int32(num_iterations, 10, "The number of iterations.");
DEFINE_int32(num_proposals_from_self, 1, "The number of proposals from self.");
DEFINE_int32(num_proposals_from_others, 1, "The number of proposals from others.");

using namespace std;
using namespace cv;
using namespace flow_fusion;


int main(int argc, char *argv[])
{
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_log_dir = "log";
  //FLAGS_logtostderr = false;
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << FLAGS_dataset_name;

  time_t timer;
  time(&timer);
  struct tm today = {0};
  today.tm_hour = today.tm_min = today.tm_sec = 0;
  today.tm_mon = 2;
  today.tm_mday = 9;
  today.tm_year = 116;
  LOG(INFO) << difftime(timer, mktime(&today)) << '\t' << -1 << '\t' << -1 << '\t' << 0 << endl;
  
  srand(time(0));
  //  PipelineParams pipeline_params(FLAGS_num_threads, FLAGS_num_fusion_iterations);

  Mat image_1 = imread(FLAGS_dataset_name + "/frame10.png");
  Mat image_2 = imread(FLAGS_dataset_name + "/frame11.png");

  const int IMAGE_WIDTH = image_1.cols;
  const int IMAGE_HEIGHT = image_1.rows;
  
  // Mat blurred_image_1, blurred_image_2;
  // GaussianBlur(image_1, blurred_image_1, cv::Size(11, 11), 0, 0);
  // GaussianBlur(image_2, blurred_image_2, cv::Size(11, 11), 0, 0);
  //subtract(image_1, blurred_image_1, image_1);
  //subtract(image_2, blurred_image_2, image_2);
  //Mat gray_image_1, gray_image_2;
  // cvtColor(image_1, gray_image_1, CV_BGR2GRAY);
  // cvtColor(image_2, gray_image_2, CV_BGR2GRAY);
  // cvtColor(gray_image_1, image_1, CV_GRAY2BGR);
  // cvtColor(gray_image_2, image_2, CV_GRAY2BGR);
  //GaussianBlur(image_1, image_1, cv::Size(5, 5), 0, 0);
  //GaussianBlur(image_2, image_2, cv::Size(5, 5), 0, 0);


  typedef ParallelFusion::LabelSpace<pair<double, double> > LABELSPACE;

  
  ParallelFusion::ParallelFusionOption option;
  option.num_threads = FLAGS_num_threads;
  option.max_iteration = FLAGS_num_iterations;
  option.selectionMethod = ParallelFusion::ParallelFusionOption::BEST;
  //option.synchronize = true;
  
  
  vector<shared_ptr<ParallelFusion::ProposalGenerator<LABELSPACE> > > generators((size_t)option.num_threads);
  vector<shared_ptr<ParallelFusion::FusionSolver<LABELSPACE> > > solvers((size_t)option.num_threads);
  vector<LABELSPACE> initials((size_t)option.num_threads);
  vector<ParallelFusion::ThreadOption> thread_options((size_t)option.num_threads);

  for (auto i = 0; i < option.num_threads; ++i) {
    generators[i] = shared_ptr<ParallelFusion::ProposalGenerator<LABELSPACE> >(new OpticalFlowProposalGenerator(image_1, image_2));
    solvers[i] = shared_ptr<ParallelFusion::FusionSolver<LABELSPACE> >(new OpticalFlowFusionSolver(image_1, image_2));
    initials[i].setSingleLabels(vector<pair<double, double> >(IMAGE_WIDTH * IMAGE_HEIGHT));
    
    thread_options[i].kSelfThread = FLAGS_num_proposals_from_self;
    thread_options[i].kOtherThread = FLAGS_num_proposals_from_others;
    // if (i == option.num_threads - 1 && false) {
    //   thread_options[i].kSelfThread = 0;
    //   thread_options[i].kOtherThread = 4;
    //   thread_options[i].is_monitor = true;
    // }
  }
  
  ParallelFusion::ParallelFusionPipeline<LABELSPACE> parallelFusionPipeline(option);
  
  float t = cv::getTickCount();
  printf("Start runing parallel optimization\n");

  //LOG(INFO) << clock() / CLOCKS_PER_SEC << '\t' << -1 << '\t' << -1 << '\t' << numeric_limits<double>::max() << endl;
  parallelFusionPipeline.runParallelFusion(initials, generators, solvers, thread_options);
  t = ((float)getTickCount() - t) / (float)getTickFrequency();

  ParallelFusion::SolutionType<LABELSPACE> solution;
  parallelFusionPipeline.getBestLabeling(solution);
  printf("Done! Final energy: %.5f\n", solution.first);
  LABELSPACE solution_label_space = solution.second;


  // {
  //   vector<pair<double, double> > flows = readFlows(FLAGS_dataset_name + "/flow10.flo");
  //   imwrite(FLAGS_dataset_name + "/flow_image.png", drawFlows(flows, IMAGE_WIDTH, IMAGE_HEIGHT));
  //   exit(1);
  // }
  
  // if (ifstream(FLAGS_dataset_name + "/flow10.flo"))
  //   return 0;
  
  
  // OpticalFlowFusionSolver fusion_solver(image_1, image_2);
  // vector<pair<double, double> > ground_truth_flows = FLAGS_evaluation ? vector<pair<double, double> >(IMAGE_WIDTH * IMAGE_HEIGHT) : readFlows("other-gt-flow/Grove2/flow10.flo");
  
  // if (FLAGS_evaluation == false) {
  //   //vector<pair<double, double> > ground_truth_flows = readFlows(FLAGS_dataset_name + "/flow10.flo");
  //   // writeFlows(ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT, "Test/flow_test.flo");
  //   // vector<pair<double, double> > test_flows = readFlows("Test/flow_test.flo");
  //   // cout << calcFlowsDiff(ground_truth_flows, test_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //   // exit(1);
    
  //   double ground_truth_energy = fusion_solver.evaluateEnergy(ground_truth_flows);
  //   cout << "ground truth energy: " << ground_truth_energy << endl;
  //   if (FLAGS_write_log)
  //     LOG(INFO) << "ground truth energy: " << ground_truth_energy << endl;
  //   //exit(1);
  
  //   imwrite("Test/Comparison/ground_truth.png", drawFlows(ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT));
  //   {
  //     vector<pair<double, double> > flows = calcFlowsNearestNeighbor(image_1, image_2, 5);
  //     cout << "NearestNeighbor error: " << calcFlowsDiff(flows, ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //     cout << "NearestNeighbor energy: " << fusion_solver.evaluateEnergy(flows) << endl;

  //     if (FLAGS_write_log) {
  //       LOG(INFO) << "NearestNeighbor error: " << calcFlowsDiff(flows, ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //       LOG(INFO) << "NearestNeighbor energy: " << fusion_solver.evaluateEnergy(flows) << endl;
  //     }
    
  //     imwrite("Test/Comparison/NearestNeighbor.png", drawFlows(flows, IMAGE_WIDTH, IMAGE_HEIGHT));
  //   }
  //   {
  //     vector<pair<double, double> > flows = calcFlowsPyrLK(image_1, image_2, 3);
  //     cout << "PyrLK error: " << calcFlowsDiff(flows, ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //     cout << "PyrLK energy: " << fusion_solver.evaluateEnergy(flows) << endl;

  //     if (FLAGS_write_log) {
  // 	LOG(INFO) << "PyrLK error: " << calcFlowsDiff(flows, ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  // 	LOG(INFO) << "PyrLK energy: " << fusion_solver.evaluateEnergy(flows) << endl;
  //     }
    
  //     imwrite("Test/Comparison/PyrLK.png", drawFlows(flows, IMAGE_WIDTH, IMAGE_HEIGHT));
  //   }
  //   {
  //     vector<pair<double, double> > flows = calcFlowsFarneback(image_1, image_2, 3, 5, 0);
  //     cout << "Farneback error: " << calcFlowsDiff(flows, ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //     cout << "Farneback energy: " << fusion_solver.evaluateEnergy(flows) << endl;

  //     if (FLAGS_write_log) {
  // 	LOG(INFO) << "Farneback error: " << calcFlowsDiff(flows, ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  // 	LOG(INFO) << "Farneback energy: " << fusion_solver.evaluateEnergy(flows) << endl;
  //     }
    
  //     imwrite("Test/Comparison/Farneback.png", drawFlows(flows, IMAGE_WIDTH, IMAGE_HEIGHT));
  //   }
  //   {
  //     vector<pair<double, double> > flows = calcFlowsLayerWise(image_1, image_2, 0.01, 0.75);
  //     cout << "LayerWise error: " << calcFlowsDiff(flows, ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //     cout << "LayerWise energy: " << fusion_solver.evaluateEnergy(flows) << endl;

  //     if (FLAGS_write_log) {
  // 	LOG(INFO) << "LayerWise error: " << calcFlowsDiff(flows, ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  // 	LOG(INFO) << "LayerWise energy: " << fusion_solver.evaluateEnergy(flows) << endl;
  //     }
    
  //     imwrite("Test/Comparison/LayerWise.png", drawFlows(flows, IMAGE_WIDTH, IMAGE_HEIGHT));
  //   }
  // }
  
  // // {
  // //   vector<pair<double, double> > flows = calcFlowsBrox(image_1, image_2, 0.01, 0.75);
  // //   cout << "Brox error: " << calcFlowsDiff(flows, ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  // //   cout << "Brox energy: " << fusion_solver.evaluateEnergy(flows) << endl;
  // // }
  
  
  // // {
  // //   LabelSpace<pair<double, double> > label_space(ground_truth_flows);
  // //   double energy;
  // //   vector<pair<double, double> > solution = fusion_solver.solve(label_space, energy);
  // //   exit(1);
  // // }
  // // for (int c = 0; c < 100; c++) {
  // //   LabelSpace<pair<double, double> > label_space(ground_truth_flows);
  // //   label_space += LabelSpace<pair<double, double> >(disturbFlows(ground_truth_flows, image_1.cols, image_1.rows, 0.1));
  // //   double energy;
  // //   vector<pair<double, double> > solution = fusion_solver.solve(label_space, energy);
  // //   double checked_energy  = fusion_solver.evaluateEnergy(solution);
  // //   cout << energy << '\t' << checked_energy << endl;
  // //   if (energy < ground_truth_energy)
  // //     cout << c << '\t' << energy << '\t' << checked_energy << endl;
  // // }
  // // exit(1);
  // // cout << "energy: " << fusion_solver.evaluateEnergy(flows) << endl;
  // // cout << "energy: " << fusion_solver.evaluateEnergy(vector<pair<double, double> >(image_1.cols * image_1.rows, make_pair(0, 0))) << endl;
  // // exit(1);
  
  // // for (int pixel = 0; pixel < image_1.cols * image_1.rows; pixel++)
  // //   for (int c = 0; c < 2; c++)
  // //     cout << flows[pixel][c] << '\t' << ground_truth_flows[pixel][c] << endl;
  // //imwrite("Test/flow_image.png", drawFlows(flows, image_1.cols, image_1.rows, 5));
  // // for (int c = 0; c < 100; c++)
  // //   cout << cv_utils::drawFromNormalDistribution(0, 0.1) << endl;
  // // exit(1);

  // //PipelineParams pipeline_params(FLAGS_num_threads, FLAGS_num_iterations, FLAGS_num_proposed_solutions);
  // vector<double> fusion_thread_master_likelihood_vec(pipeline_params.NUM_THREADS, 0);
  // //fusion_thread_master_likelihood_vec[pipeline_params.NUM_THREADS - 1] = 1;
  // vector<unique_ptr<FusionThread<pair<double, double> > > > fusion_threads;
  // vector<shared_ptr<ProposalGenerator<pair<double, double> > > > proposal_generators;
  // for (int fusion_thread_index = 0; fusion_thread_index < pipeline_params.NUM_THREADS; fusion_thread_index++)
  //   proposal_generators.push_back(dynamic_pointer_cast<ProposalGenerator<pair<double, double> > >(shared_ptr<OpticalFlowProposalGenerator>(new OpticalFlowProposalGenerator(image_1, image_2, pipeline_params.NUM_PROPOSAL_SOLUTIONS_THRESHOLD, ground_truth_flows))));
  // vector<shared_ptr<FusionSolver<pair<double, double> > > > fusion_solvers;
  // for (int fusion_thread_index = 0; fusion_thread_index < pipeline_params.NUM_THREADS; fusion_thread_index++)
  //   fusion_solvers.push_back(dynamic_pointer_cast<FusionSolver<pair<double, double> > >(shared_ptr<OpticalFlowFusionSolver>(new OpticalFlowFusionSolver(image_1, image_2))));
  // for (int fusion_thread_index = 0; fusion_thread_index < pipeline_params.NUM_THREADS; fusion_thread_index++)			  
  //   fusion_threads.push_back(move(unique_ptr<FusionThread<pair<double, double> > >(new FusionThread<pair<double, double> >(proposal_generators[fusion_thread_index], fusion_solvers[fusion_thread_index], fusion_thread_master_likelihood_vec[fusion_thread_index]))));

  // vector<pair<double, double> > initial_solution(IMAGE_WIDTH * IMAGE_HEIGHT);
  // disturbFlows(initial_solution, IMAGE_WIDTH, IMAGE_HEIGHT, 3);
  // //vector<pair<double, double> > initial_solution = ground_truth_flows;

  // const clock_t begin_time = clock();
  // vector<pair<double, double> > fused_solution = parallelFuse(fusion_threads, pipeline_params, initial_solution);
  // double running_time = static_cast<double>((clock() - begin_time) / CLOCKS_PER_SEC);
    
  // //  cout << FLAGS_dataset_name << endl;
  // cout << "final error: " << calcFlowsDiff(fused_solution, ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;;
  // cout << "running time: " << running_time << endl;

  // if (FLAGS_write_log) {
  //   LOG(INFO) << "final error: " << calcFlowsDiff(fused_solution, ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;;
  //   LOG(INFO) << "final energy: " << fusion_solver.evaluateEnergy(fused_solution) << endl;;
  //   LOG(INFO) << "running time: " << running_time << endl;
  // }
  
  // writeFlows(fused_solution, IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS_dataset_name + "/flow10.flo");
  // return 0;
}
