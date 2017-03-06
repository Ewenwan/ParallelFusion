#include <gflags/gflags.h>

#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <vector>

#include "../base/HFusionPipeline.h"
#include "../base/ParallelFusionPipeline.h"
//#include "../base/cv_utils/cv_utils.h"
#include "OpticalFlowCalculation.h"
#include "OpticalFlowFusionSolver.h"
#include "OpticalFlowProposalGenerator.h"
#include "OpticalFlowUtils.h"

// DEFINE_int32(num_fusion_iterations, 1, "The number of iterations in fusion
// pipeline.");
// DEFINE_int32(num_threads, 1, "The number of threads.");

// DEFINE_string(dataset_name, "other-data/Dimetrodon", "The dataset image
// name.");
DEFINE_string(dataset_name, "other-data/Dimetrodon", "The dataset image name.");
DEFINE_int32(num_proposed_solutions, 4,
             "The number of proposed solution for each thread.");
DEFINE_bool(write_log, true, "Write log file or not.");
DEFINE_bool(evaluation, false, "Write log file or not.");
// DEFINE_string(dataset_category, "other-data", "The dataset image category.");

DEFINE_int32(num_threads, 4, "The number of threads.");
DEFINE_int32(num_iterations, 30, "The number of iterations.");

DEFINE_int32(num_proposals_in_total, 1, "The number of proposals in total.");
DEFINE_int32(num_proposals_from_others, 0,
             "The number of proposals from others.");
DEFINE_int32(
    solution_exchange_interval, 3,
    "The number of iterations between consecutive solution exchanges.");
DEFINE_int32(result_index, 0, "The index of the result.");
DEFINE_bool(use_monitor_thread, false, "Whether monitor object is used.");

using namespace std;
using namespace cv;
using namespace flow_fusion;

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  FLAGS_log_dir = "Log";
  // FLAGS_logtostderr = false;
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << FLAGS_dataset_name;
  LOG(INFO) << FLAGS_num_threads << '\t' << FLAGS_num_iterations << '\t'
            << FLAGS_num_proposals_in_total << '\t'
            << FLAGS_num_proposals_from_others << '\t'
            << FLAGS_solution_exchange_interval << '\t'
            << FLAGS_use_monitor_thread << endl;

  time_t timer;
  time(&timer);
  struct tm today = {0};
  today.tm_hour = today.tm_min = today.tm_sec = 0;
  today.tm_mon = 2;
  today.tm_mday = 9;
  today.tm_year = 116;
  int start_time = difftime(timer, mktime(&today));
  LOG(INFO) << start_time << '\t' << -1 << '\t' << -1 << '\t' << 0 << endl;

  // srand(0);
  //  PipelineParams pipeline_params(FLAGS_num_threads,
  //  FLAGS_num_fusion_iterations);

  Mat image_1 = imread(FLAGS_dataset_name + "/frame10.png");
  Mat image_2 = imread(FLAGS_dataset_name + "/frame11.png");

  CHECK_NOTNULL(image_1.data);
  CHECK_NOTNULL(image_2.data);

  const int IMAGE_WIDTH = image_1.cols;
  const int IMAGE_HEIGHT = image_1.rows;

  // Mat blurred_image_1, blurred_image_2;
  // GaussianBlur(image_1, blurred_image_1, cv::Size(11, 11), 0, 0);
  // GaussianBlur(image_2, blurred_image_2, cv::Size(11, 11), 0, 0);
  // subtract(image_1, blurred_image_1, image_1);
  // subtract(image_2, blurred_image_2, image_2);
  // Mat gray_image_1, gray_image_2;
  // cvtColor(image_1, gray_image_1, CV_BGR2GRAY);
  // cvtColor(image_2, gray_image_2, CV_BGR2GRAY);
  // cvtColor(gray_image_1, image_1, CV_GRAY2BGR);
  // cvtColor(gray_image_2, image_2, CV_GRAY2BGR);
  // GaussianBlur(image_1, image_1, cv::Size(5, 5), 0, 0);
  // GaussianBlur(image_2, image_2, cv::Size(5, 5), 0, 0);

  vector<pair<double, double>> ground_truth_flows =
      FLAGS_evaluation
          ? readFlows("other-gt-flow/Dimetrodon/flow10.flo")
          : vector<pair<double, double>>(IMAGE_WIDTH * IMAGE_HEIGHT);

#if 0
    typedef ParallelFusion::LabelSpace<pair<double, double>> LABELSPACE;
    ParallelFusion::HFusionPipelineOption option;
    option.num_threads = FLAGS_num_threads;
    // option.synchronize = true;

    OpticalFlowProposalGenerator generator(image_1, image_2,
                                           ground_truth_flows);
    vector<LABELSPACE> all_proposals;
    generator.getAllProposals(all_proposals);
    vector<shared_ptr<ParallelFusion::FusionSolver<LABELSPACE>>> solvers(
        (size_t)option.num_threads);
    vector<ParallelFusion::ThreadOption> thread_options(
        (size_t)option.num_threads);

    for (auto i = 0; i < option.num_threads; ++i) {
      solvers[i] = shared_ptr<ParallelFusion::FusionSolver<LABELSPACE>>(
          new OpticalFlowFusionSolver(image_1, image_2));
    }

    ParallelFusion::HFusionPipeline<LABELSPACE> h_fusion_pipeline(option);
    h_fusion_pipeline.runHFusion(
        all_proposals, solvers,
        LABELSPACE(vector<pair<double, double>>(IMAGE_WIDTH * IMAGE_HEIGHT)));

    vector<ParallelFusion::Observation> observations =
        h_fusion_pipeline.getGlobalProfile().getProfile();
    for (vector<ParallelFusion::Observation>::const_iterator observation_it =
             observations.begin();
         observation_it != observations.end(); observation_it++) {
      cout << observation_it->first << '\t' << observation_it->second << endl;
    }
    exit(1);
#endif

  typedef ParallelFusion::LabelSpace<pair<double, double>> LABELSPACE;

  ParallelFusion::ParallelFusionOption option;
  option.num_threads = FLAGS_num_threads;
  option.max_iteration = FLAGS_num_iterations;
  option.selectionMethod = ParallelFusion::ParallelFusionOption::RANDOM;
  option.timeout = std::chrono::minutes(5);
  // option.synchronize = true;

  vector<shared_ptr<ParallelFusion::ProposalGenerator<LABELSPACE>>> generators(
      (size_t)option.num_threads);
  vector<shared_ptr<ParallelFusion::FusionSolver<LABELSPACE>>> solvers(
      (size_t)option.num_threads);
  vector<LABELSPACE> initials((size_t)option.num_threads);
  vector<ParallelFusion::ThreadOption> thread_options(
      (size_t)option.num_threads);

  for (auto i = 0; i < option.num_threads; ++i) {
    generators[i] = shared_ptr<ParallelFusion::ProposalGenerator<LABELSPACE>>(
        new OpticalFlowProposalGenerator(image_1, image_2, ground_truth_flows));
    solvers[i] = shared_ptr<ParallelFusion::FusionSolver<LABELSPACE>>(
        new OpticalFlowFusionSolver(image_1, image_2, option.timeout));
    initials[i].setSingleLabels(
        vector<pair<double, double>>(IMAGE_WIDTH * IMAGE_HEIGHT));

    thread_options[i].kTotal = FLAGS_num_proposals_in_total;
    thread_options[i].kOtherThread = FLAGS_num_proposals_from_others;
    thread_options[i].solution_exchange_interval =
        FLAGS_solution_exchange_interval;

    if (i == option.num_threads - 1 && FLAGS_use_monitor_thread) {
      thread_options[i].kTotal = 1;
      // thread_options[i].kOtherThread = 2;
      // thread_options[i].solution_exchange_interval = 1;
      thread_options[i].is_monitor = true;
    }
  }

// for (auto i = 0; i < option.num_threads - FLAGS_use_monitor_thread; ++i) {
//   imwrite("Test/solution_image_" + to_string(i) + "_0.png",
//   drawFlows(vector<pair<double, double> >(IMAGE_WIDTH * IMAGE_HEIGHT),
//   IMAGE_WIDTH, IMAGE_HEIGHT));
// }

#if 0
    const string solution_path = "Test/Victor/";
    const int BORDER_WIDTH = 10;
    const int FPS = 24;
    Rect flow_ROI(BORDER_WIDTH, BORDER_WIDTH, IMAGE_WIDTH - BORDER_WIDTH * 2,
                  IMAGE_HEIGHT - BORDER_WIDTH * 2);
    vector<vector<Solution>> thread_solutions(option.num_threads -
                                              FLAGS_use_monitor_thread);
    for (auto thread_id = 0;
         thread_id < option.num_threads - FLAGS_use_monitor_thread;
         ++thread_id) {
      ifstream in_str(solution_path + "solutions_" + to_string(thread_id));
      int num_solutions = 0;
      in_str >> num_solutions;
      thread_solutions[thread_id].resize(num_solutions);
      for (int i = 0; i < num_solutions; i++)
        in_str >> thread_solutions[thread_id][i];
    }

    map<int, Solution> time_global_solution_map;
    for (auto thread_id = 0;
         thread_id < option.num_threads - FLAGS_use_monitor_thread; ++thread_id)
      for (vector<Solution>::iterator solution_it =
               thread_solutions[thread_id].begin();
           solution_it != thread_solutions[thread_id].end(); solution_it++)
        time_global_solution_map[solution_it->time] = *solution_it;

    int max_time = 0;
    double previous_energy = std::numeric_limits<double>::max();
    map<int, Solution> new_time_global_solution_map;
    for (map<int, Solution>::iterator time_solution_it =
             time_global_solution_map.begin();
         time_solution_it != time_global_solution_map.end();
         time_solution_it++) {
      if (time_solution_it->second.energy < previous_energy) {
        new_time_global_solution_map[time_solution_it->first] =
            time_solution_it->second;
        previous_energy = time_solution_it->second.energy;
        max_time = time_solution_it->first;
      }
    }
    time_global_solution_map = new_time_global_solution_map;

    max_time += 10;

    auto addText = [](const Mat &image, const string &text) {
      const int IMAGE_WIDTH = image.cols;
      const int IMAGE_HEIGHT = image.rows;
      Mat image_with_padding = image;
      // vconcat(image, Mat::zeros(100, IMAGE_WIDTH, CV_8UC3),
      // image_with_padding);
      //      copyMakeBorder(image, image_with_padding, 0, IMAGE_HEIGHT * 0.1,
      //      0, 0, BORDER_CONSTANT, Scalar(0, 0, 0));

      if (text.size() > 0)
        putText(image_with_padding, text,
                Point(IMAGE_WIDTH / 10, IMAGE_HEIGHT / 2), FONT_HERSHEY_PLAIN,
                3, Scalar(0, 0, 255));
      return image_with_padding;
    };

    for (auto thread_id = 0;
         thread_id < option.num_threads - FLAGS_use_monitor_thread;
         ++thread_id) {
      map<int, Solution> time_solution_map;
      for (vector<Solution>::iterator solution_it =
               thread_solutions[thread_id].begin();
           solution_it != thread_solutions[thread_id].end(); solution_it++)
        time_solution_map[solution_it->time] = *solution_it;

      VideoWriter video_writer(solution_path + "solution_video_" +
                                   to_string(thread_id) + ".mp4",
                               VideoWriter::fourcc('X', '2', '6', '4'), FPS,
                               Size(IMAGE_WIDTH - BORDER_WIDTH * 2,
                                    IMAGE_HEIGHT - BORDER_WIDTH * 2));
      if (video_writer.isOpened()) {
        Mat previous_image =
            drawFlows(vector<pair<double, double>>(IMAGE_WIDTH * IMAGE_HEIGHT),
                      IMAGE_WIDTH, IMAGE_HEIGHT);
        previous_image = previous_image(flow_ROI);
        // previous_image = addText(previous_image, "");
        for (int frame = 0; frame < max_time * FPS; frame++) {
          int time = frame / FPS;
          if (time_solution_map.count(time) > 0) {
            Mat image =
                imread(solution_path + "solution_image_" +
                       to_string(thread_id) + "_" + to_string(time) + ".png");
            string text =
                to_string(time_solution_map[time].thread_id) + "  " +
                to_string(static_cast<int>(time_solution_map[time].energy));
            for (std::vector<int>::const_iterator thread_it =
                     time_solution_map[time].selected_threads.begin();
                 thread_it != time_solution_map[time].selected_threads.end();
                 thread_it++)
              text += "  " + to_string(*thread_it);

            image = image(flow_ROI);
            // image = addText(image, text);
            previous_image = image;
          }
          // Mat test = previous_image(flow_ROI);
          // cout << test.size() << '\t' << previous_image.size() << endl;
          // exit(1);
          // cout << previous_image.cols << '\t' << previous_image.rows << endl;
          video_writer << previous_image;
        }
      } else {
        cout << "Cannot open video file." << endl;
      }
    }

    {
      VideoWriter video_writer(solution_path + "solution_video_global.mp4",
                               VideoWriter::fourcc('X', '2', '6', '4'), FPS,
                               Size(IMAGE_WIDTH - BORDER_WIDTH * 2,
                                    IMAGE_HEIGHT - BORDER_WIDTH * 2));
      if (video_writer.isOpened()) {
        Mat previous_image =
            drawFlows(vector<pair<double, double>>(IMAGE_WIDTH * IMAGE_HEIGHT),
                      IMAGE_WIDTH, IMAGE_HEIGHT);
        previous_image = addText(previous_image(flow_ROI), "");
        for (int frame = 0; frame < max_time * FPS; frame++) {
          int time = frame / FPS;
          if (time_global_solution_map.count(time) > 0) {
            Mat image =
                imread(solution_path + "solution_image_" +
                       to_string(time_global_solution_map[time].thread_id) +
                       "_" + to_string(time) + ".png");
            string text = to_string(time_global_solution_map[time].thread_id) +
                          "  " + to_string(static_cast<int>(
                                     time_global_solution_map[time].energy));

            image = addText(image(flow_ROI), text);
            previous_image = image;
          }
          // Mat test = previous_image(flow_ROI);
          // cout << test.size() << '\t' << previous_image.size() << endl;
          // exit(1);
          video_writer << previous_image;
        }
      } else {
        cout << "Cannot open video file." << endl;
      }
    }
    return 0;
#endif

  ParallelFusion::ParallelFusionPipeline<LABELSPACE> parallelFusionPipeline(
      option);

  float t = cv::getTickCount();
  printf("Start runing parallel optimization\n");

  // LOG(INFO) << clock() / CLOCKS_PER_SEC << '\t' << -1 << '\t' << -1 << '\t'
  // << numeric_limits<double>::max() << endl;
  parallelFusionPipeline.runParallelFusion(initials, generators, solvers,
                                           thread_options);
  t = ((float)getTickCount() - t) / (float)getTickFrequency();

  ParallelFusion::SolutionType<LABELSPACE> solution;
  parallelFusionPipeline.getBestLabeling(solution);
  printf("Done! Final energy: %.5f\n", solution.first);
  LABELSPACE solution_label_space = solution.second;

  const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
  vector<pair<double, double>> solution_labels(NUM_PIXELS);
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
    solution_labels[pixel] = solution_label_space.getLabelOfNode(pixel)[0];

  imwrite("Results/flow_image_" + to_string(FLAGS_result_index) + ".png",
          drawFlows(solution_labels, IMAGE_WIDTH, IMAGE_HEIGHT));
  writeFlows(solution_labels, IMAGE_WIDTH, IMAGE_HEIGHT,
             "Results/flow_" + to_string(FLAGS_result_index) + ".flo");

  const int BORDER_WIDTH = 10;
  Rect flow_ROI(BORDER_WIDTH, BORDER_WIDTH, IMAGE_WIDTH - BORDER_WIDTH * 2,
                IMAGE_HEIGHT - BORDER_WIDTH * 2);

  for (auto thread_id = 0;
       thread_id < option.num_threads - FLAGS_use_monitor_thread; ++thread_id) {
    vector<Solution> solutions =
        dynamic_pointer_cast<OpticalFlowProposalGenerator>(
            generators[thread_id])
            ->getSolutions();

    ofstream out_str("Test/solutions_" + to_string(thread_id));
    out_str << solutions.size() << endl << endl;
    for (vector<Solution>::iterator solution_it = solutions.begin();
         solution_it != solutions.end(); solution_it++) {
      solution_it->time -= start_time;
      out_str << *solution_it << endl;
      imwrite(
          "Test/solution_image_" + to_string(solution_it->thread_id) + "_" +
              to_string(solution_it->time) + ".png",
          drawFlows(solution_it->solution_labels, IMAGE_WIDTH, IMAGE_HEIGHT));
    }

    // int fps = 4;
    // VideoWriter video_writer("Test/solution_video_" + to_string(thread_id) +
    // ".avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(IMAGE_WIDTH -
    // BORDER_WIDTH * 2, IMAGE_HEIGHT - BORDER_WIDTH * 2));
    // Mat previous_image = drawFlows(vector<pair<double, double> >(IMAGE_WIDTH
    // * IMAGE_HEIGHT), IMAGE_WIDTH, IMAGE_HEIGHT);

    // int num_written_solutions = 0;
    // for (int time_diff = 0; ; time_diff++) {
    //   int time = start_time + time_diff;
    //   if (time_solution_map.count(time) > 0) {
    //     previous_image = drawFlows(time_solution_map[time], IMAGE_WIDTH,
    //     IMAGE_HEIGHT);
    //     //imwrite("Test/solution_image_" + to_string(thread_index) + "_" +
    //     to_string(time_diff) + ".png", solution_image);
    //     num_written_solutions++;
    //   }
    //   video_writer << previous_image(flow_ROI);
    //   if (num_written_solutions == time_solution_map.size())
    //     break;
    // }
  }

  // {
  //   vector<pair<double, double> > flows = readFlows(FLAGS_dataset_name +
  //   "/flow10.flo");
  //   imwrite(FLAGS_dataset_name + "/flow_image.png", drawFlows(flows,
  //   IMAGE_WIDTH, IMAGE_HEIGHT));
  //   exit(1);
  // }

  // if (ifstream(FLAGS_dataset_name + "/flow10.flo"))
  //   return 0;

  // OpticalFlowFusionSolver fusion_solver(image_1, image_2);
  // vector<pair<double, double> > ground_truth_flows = FLAGS_evaluation ?
  // vector<pair<double, double> >(IMAGE_WIDTH * IMAGE_HEIGHT) :
  // readFlows("other-gt-flow/Grove2/flow10.flo");

  // if (FLAGS_evaluation == false) {
  //   //vector<pair<double, double> > ground_truth_flows =
  //   readFlows(FLAGS_dataset_name + "/flow10.flo");
  //   // writeFlows(ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT,
  //   "Test/flow_test.flo");
  //   // vector<pair<double, double> > test_flows =
  //   readFlows("Test/flow_test.flo");
  //   // cout << calcFlowsDiff(ground_truth_flows, test_flows, IMAGE_WIDTH,
  //   IMAGE_HEIGHT) << endl;
  //   // exit(1);

  //   double ground_truth_energy =
  //   fusion_solver.evaluateEnergy(ground_truth_flows);
  //   cout << "ground truth energy: " << ground_truth_energy << endl;
  //   if (FLAGS_write_log)
  //     LOG(INFO) << "ground truth energy: " << ground_truth_energy << endl;
  //   //exit(1);

  //   imwrite("Test/Comparison/ground_truth.png", drawFlows(ground_truth_flows,
  //   IMAGE_WIDTH, IMAGE_HEIGHT));
  //   {
  //     vector<pair<double, double> > flows = calcFlowsNearestNeighbor(image_1,
  //     image_2, 5);
  //     cout << "NearestNeighbor error: " << calcFlowsDiff(flows,
  //     ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //     cout << "NearestNeighbor energy: " <<
  //     fusion_solver.evaluateEnergy(flows) << endl;

  //     if (FLAGS_write_log) {
  //       LOG(INFO) << "NearestNeighbor error: " << calcFlowsDiff(flows,
  //       ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //       LOG(INFO) << "NearestNeighbor energy: " <<
  //       fusion_solver.evaluateEnergy(flows) << endl;
  //     }

  //     imwrite("Test/Comparison/NearestNeighbor.png", drawFlows(flows,
  //     IMAGE_WIDTH, IMAGE_HEIGHT));
  //   }
  //   {
  //     vector<pair<double, double> > flows = calcFlowsPyrLK(image_1, image_2,
  //     3);
  //     cout << "PyrLK error: " << calcFlowsDiff(flows, ground_truth_flows,
  //     IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //     cout << "PyrLK energy: " << fusion_solver.evaluateEnergy(flows) <<
  //     endl;

  //     if (FLAGS_write_log) {
  //  LOG(INFO) << "PyrLK error: " << calcFlowsDiff(flows, ground_truth_flows,
  // IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //  LOG(INFO) << "PyrLK energy: " << fusion_solver.evaluateEnergy(flows) <<
  // endl;
  //     }

  //     imwrite("Test/Comparison/PyrLK.png", drawFlows(flows, IMAGE_WIDTH,
  //     IMAGE_HEIGHT));
  //   }
  //   {
  //     vector<pair<double, double> > flows = calcFlowsFarneback(image_1,
  //     image_2, 3, 5, 0);
  //     cout << "Farneback error: " << calcFlowsDiff(flows, ground_truth_flows,
  //     IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //     cout << "Farneback energy: " << fusion_solver.evaluateEnergy(flows) <<
  //     endl;

  //     if (FLAGS_write_log) {
  //  LOG(INFO) << "Farneback error: " << calcFlowsDiff(flows,
  // ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //  LOG(INFO) << "Farneback energy: " << fusion_solver.evaluateEnergy(flows)
  // << endl;
  //     }

  //     imwrite("Test/Comparison/Farneback.png", drawFlows(flows, IMAGE_WIDTH,
  //     IMAGE_HEIGHT));
  //   }
  //   {
  //     vector<pair<double, double> > flows = calcFlowsLayerWise(image_1,
  //     image_2, 0.01, 0.75);
  //     cout << "LayerWise error: " << calcFlowsDiff(flows, ground_truth_flows,
  //     IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //     cout << "LayerWise energy: " << fusion_solver.evaluateEnergy(flows) <<
  //     endl;

  //     if (FLAGS_write_log) {
  //  LOG(INFO) << "LayerWise error: " << calcFlowsDiff(flows,
  // ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  //  LOG(INFO) << "LayerWise energy: " << fusion_solver.evaluateEnergy(flows)
  // << endl;
  //     }

  //     imwrite("Test/Comparison/LayerWise.png", drawFlows(flows, IMAGE_WIDTH,
  //     IMAGE_HEIGHT));
  //   }
  // }

  // // {
  // //   vector<pair<double, double> > flows = calcFlowsBrox(image_1, image_2,
  // 0.01, 0.75);
  // //   cout << "Brox error: " << calcFlowsDiff(flows, ground_truth_flows,
  // IMAGE_WIDTH, IMAGE_HEIGHT) << endl;
  // //   cout << "Brox energy: " << fusion_solver.evaluateEnergy(flows) <<
  // endl;
  // // }

  // // {
  // //   LabelSpace<pair<double, double> > label_space(ground_truth_flows);
  // //   double energy;
  // //   vector<pair<double, double> > solution =
  // fusion_solver.solve(label_space, energy);
  // //   exit(1);
  // // }
  // // for (int c = 0; c < 100; c++) {
  // //   LabelSpace<pair<double, double> > label_space(ground_truth_flows);
  // //   label_space += LabelSpace<pair<double, double>
  // >(disturbFlows(ground_truth_flows, image_1.cols, image_1.rows, 0.1));
  // //   double energy;
  // //   vector<pair<double, double> > solution =
  // fusion_solver.solve(label_space, energy);
  // //   double checked_energy  = fusion_solver.evaluateEnergy(solution);
  // //   cout << energy << '\t' << checked_energy << endl;
  // //   if (energy < ground_truth_energy)
  // //     cout << c << '\t' << energy << '\t' << checked_energy << endl;
  // // }
  // // exit(1);
  // // cout << "energy: " << fusion_solver.evaluateEnergy(flows) << endl;
  // // cout << "energy: " << fusion_solver.evaluateEnergy(vector<pair<double,
  // double> >(image_1.cols * image_1.rows, make_pair(0, 0))) << endl;
  // // exit(1);

  // // for (int pixel = 0; pixel < image_1.cols * image_1.rows; pixel++)
  // //   for (int c = 0; c < 2; c++)
  // //     cout << flows[pixel][c] << '\t' << ground_truth_flows[pixel][c] <<
  // endl;
  // //imwrite("Test/flow_image.png", drawFlows(flows, image_1.cols,
  // image_1.rows, 5));
  // // for (int c = 0; c < 100; c++)
  // //   cout << cv_utils::drawFromNormalDistribution(0, 0.1) << endl;
  // // exit(1);

  // //PipelineParams pipeline_params(FLAGS_num_threads, FLAGS_num_iterations,
  // FLAGS_num_proposed_solutions);
  // vector<double>
  // fusion_thread_master_likelihood_vec(pipeline_params.NUM_THREADS, 0);
  // //fusion_thread_master_likelihood_vec[pipeline_params.NUM_THREADS - 1] = 1;
  // vector<unique_ptr<FusionThread<pair<double, double> > > > fusion_threads;
  // vector<shared_ptr<ProposalGenerator<pair<double, double> > > >
  // proposal_generators;
  // for (int fusion_thread_index = 0; fusion_thread_index <
  // pipeline_params.NUM_THREADS; fusion_thread_index++)
  //   proposal_generators.push_back(dynamic_pointer_cast<ProposalGenerator<pair<double,
  //   double> > >(shared_ptr<OpticalFlowProposalGenerator>(new
  //   OpticalFlowProposalGenerator(image_1, image_2,
  //   pipeline_params.NUM_PROPOSAL_SOLUTIONS_THRESHOLD, ground_truth_flows))));
  // vector<shared_ptr<FusionSolver<pair<double, double> > > > fusion_solvers;
  // for (int fusion_thread_index = 0; fusion_thread_index <
  // pipeline_params.NUM_THREADS; fusion_thread_index++)
  //   fusion_solvers.push_back(dynamic_pointer_cast<FusionSolver<pair<double,
  //   double> > >(shared_ptr<OpticalFlowFusionSolver>(new
  //   OpticalFlowFusionSolver(image_1, image_2))));
  // for (int fusion_thread_index = 0; fusion_thread_index <
  // pipeline_params.NUM_THREADS; fusion_thread_index++)
  //   fusion_threads.push_back(move(unique_ptr<FusionThread<pair<double,
  //   double> > >(new FusionThread<pair<double, double>
  //   >(proposal_generators[fusion_thread_index],
  //   fusion_solvers[fusion_thread_index],
  //   fusion_thread_master_likelihood_vec[fusion_thread_index]))));

  // vector<pair<double, double> > initial_solution(IMAGE_WIDTH * IMAGE_HEIGHT);
  // disturbFlows(initial_solution, IMAGE_WIDTH, IMAGE_HEIGHT, 3);
  // //vector<pair<double, double> > initial_solution = ground_truth_flows;

  // const clock_t begin_time = clock();
  // vector<pair<double, double> > fused_solution = parallelFuse(fusion_threads,
  // pipeline_params, initial_solution);
  // double running_time = static_cast<double>((clock() - begin_time) /
  // CLOCKS_PER_SEC);

  // //  cout << FLAGS_dataset_name << endl;
  // cout << "final error: " << calcFlowsDiff(fused_solution,
  // ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;;
  // cout << "running time: " << running_time << endl;

  // if (FLAGS_write_log) {
  //   LOG(INFO) << "final error: " << calcFlowsDiff(fused_solution,
  //   ground_truth_flows, IMAGE_WIDTH, IMAGE_HEIGHT) << endl;;
  //   LOG(INFO) << "final energy: " <<
  //   fusion_solver.evaluateEnergy(fused_solution) << endl;;
  //   LOG(INFO) << "running time: " << running_time << endl;
  // }

  // writeFlows(fused_solution, IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS_dataset_name +
  // "/flow10.flo");
  // return 0;
}
