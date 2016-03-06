#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <map>

#include "SegmentationProposalGenerator.h"
#include "SegmentationFusionSolver.h"
#include "../base/BaseDataStructures.h"
#include "../base/ParallelFusionPipeline.h"
#include "../base/FusionThread.h"

using namespace std;
using namespace cv;

int main()
{
  Mat image = imread("Inputs/toy_example.png");
   
  //parallel fusion move
  //typedef LabelType int;
  
  if (true) {
    PipelineParams pipeline_params(4, 5);
    vector<double> fusion_thread_master_likelihood_vec(pipeline_params.NUM_THREADS, 0);
    fusion_thread_master_likelihood_vec[pipeline_params.NUM_THREADS - 1] = 1;
    vector<unique_ptr<FusionThread<int> > > fusion_threads;
    vector<shared_ptr<ProposalGenerator<int> > > proposal_generators;
    for (int fusion_thread_index = 0; fusion_thread_index < pipeline_params.NUM_THREADS; fusion_thread_index++)
      proposal_generators.push_back(dynamic_pointer_cast<ProposalGenerator<int> >(shared_ptr<SegmentationProposalGenerator>(new SegmentationProposalGenerator(image))));
    vector<shared_ptr<FusionSolver<int> > > fusion_solvers;
    for (int fusion_thread_index = 0; fusion_thread_index < pipeline_params.NUM_THREADS; fusion_thread_index++)
      fusion_solvers.push_back(dynamic_pointer_cast<FusionSolver<int> >(shared_ptr<SegmentationFusionSolver>(new SegmentationFusionSolver(image))));
    for (int fusion_thread_index = 0; fusion_thread_index < pipeline_params.NUM_THREADS; fusion_thread_index++)			  
      fusion_threads.push_back(move(unique_ptr<FusionThread<int> >(new FusionThread<int>(proposal_generators[fusion_thread_index], fusion_solvers[fusion_thread_index], fusion_thread_master_likelihood_vec[fusion_thread_index]))));
    
    vector<int> fused_solution = parallelFuse(fusion_threads, pipeline_params, vector<int>(image.cols * image.rows, 0));

    {
      map<int, Vec3b> color_table;
      color_table[0] = Vec3b(255, 255, 255);
      color_table[1] = Vec3b(0, 0, 0);
      color_table[2] = Vec3b(0, 0, 255);
      color_table[3] = Vec3b(0, 255, 0);
      color_table[4] = Vec3b(255, 0, 0);

      const int IMAGE_WIDTH = image.cols;
      const int IMAGE_HEIGHT = image.rows;
      const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
      Mat solution_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
	solution_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = color_table[fused_solution[pixel]];
      }
      imwrite("Test/solution_image.png", solution_image);
    }
  }
  
  //sequential fusion move
  if (false) {
    shared_ptr<SegmentationProposalGenerator> proposal_generator(new SegmentationProposalGenerator(image));
    shared_ptr<SegmentationFusionSolver> fusion_solver(new SegmentationFusionSolver(image));
    vector<int> current_solution = vector<int>(image.cols * image.rows, 0);
    for (int iteration = 0; iteration < 10; iteration++) {
      proposal_generator->setCurrentSolution(current_solution);
      LabelSpace<int> label_space = proposal_generator->getProposal();
      double dummy;
      current_solution = fusion_solver->solve(label_space, dummy);

      {
	map<int, Vec3b> color_table;
	color_table[0] = Vec3b(255, 255, 255);
	color_table[1] = Vec3b(0, 0, 0);
	color_table[2] = Vec3b(0, 0, 255);
	color_table[3] = Vec3b(0, 255, 0);
	color_table[4] = Vec3b(255, 0, 0);

	const int IMAGE_WIDTH = image.cols;
	const int IMAGE_HEIGHT = image.rows;
	const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
	Mat solution_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
	for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
	  solution_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = color_table[current_solution[pixel]];
	}
	imwrite("Test/sequential_solution_image_" + to_string(iteration) + ".png", solution_image);
      }
    }
  }
  return 0;
}
