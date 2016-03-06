#include "ParallelFusionPipeline.h"

#include <vector>
#include <limits>
#include <iostream>


#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../base/cv_utils/cv_utils.h"

using namespace std;

using namespace cv;

vector<int> parallelFuse(vector<unique_ptr<FusionThread> > &fusion_threads, const PipelineParams &pipeline_params, const vector<int> &initial_solution)
{
  vector<vector<int> > solution_pool(pipeline_params.NUM_THREADS, initial_solution);

  vector<double> energy_vec(pipeline_params.NUM_THREADS, numeric_limits<double>::max()); //Keep only the solution with lowest energy at the moment.
  
  vector<double> solution_confidence_vec(pipeline_params.NUM_THREADS, 1);
  for (int iteration = 0; iteration < pipeline_params.NUM_FUSION_ITERATIONS; iteration++) {
    //TODO: Parallel each thread.
    cout << "iteration: " << iteration << endl;
    for (vector<unique_ptr<FusionThread> >::iterator fusion_thread_it = fusion_threads.begin(); fusion_thread_it != fusion_threads.end(); fusion_thread_it++) {
      cout << "thread: " << fusion_thread_it - fusion_threads.begin() << '\t' << (*fusion_thread_it)->getStatus() << endl;
      vector<int> current_solution = solution_pool[cv_utils::drawFromArray(solution_confidence_vec)];
      (*fusion_thread_it)->setSolutionPool(solution_pool);
      (*fusion_thread_it)->setCurrentSolution(current_solution);
      (*fusion_thread_it)->runFusion();
      solution_pool[fusion_thread_it - fusion_threads.begin()] = (*fusion_thread_it)->getFusedSolution();
      energy_vec[fusion_thread_it - fusion_threads.begin()] = (*fusion_thread_it)->getFusedSolutionEnergy();
      vector<double>::const_iterator min_it = min_element(energy_vec.begin(), energy_vec.end());
      solution_confidence_vec.assign(pipeline_params.NUM_THREADS, 0);
      solution_confidence_vec[min_it - energy_vec.begin()] = 1;

      {
	map<int, Vec3b> color_table;
        color_table[0] = Vec3b(255, 255, 255);
        color_table[1] = Vec3b(0, 0, 0);
        color_table[2] = Vec3b(0, 0, 255);
        color_table[3] = Vec3b(0, 255, 0);
        color_table[4] = Vec3b(255, 0, 0);

        vector<int> fused_solution = solution_pool[fusion_thread_it - fusion_threads.begin()];
	const int IMAGE_WIDTH = 300;
	const int IMAGE_HEIGHT = 300;
	const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
	Mat solution_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
	for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
	  solution_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = color_table[fused_solution[pixel]];
	}
	imwrite("Test/solution_image_" + to_string(iteration) + "_" + to_string(fusion_thread_it - fusion_threads.begin()) + ".png", solution_image);
      }
    }
  }

  return solution_pool[min_element(energy_vec.begin(), energy_vec.end()) - energy_vec.begin()];
}
    
