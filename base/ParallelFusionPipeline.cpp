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

template<class LabelType> vector<LabelType> parallelFuse(vector<unique_ptr<FusionThread<LabelType> > > &fusion_threads, const PipelineParams &pipeline_params, const vector<LabelType> &initial_solution)
{
  vector<vector<LabelType> > solution_pool(pipeline_params.NUM_THREADS, initial_solution);

  vector<double> energy_vec(pipeline_params.NUM_THREADS, numeric_limits<double>::max()); //Keep only the solution with lowest energy at the moment.
  
  vector<double> solution_confidence_vec(pipeline_params.NUM_THREADS, 1);
  for (int iteration = 0; iteration < pipeline_params.NUM_FUSION_ITERATIONS; iteration++) {
    //TODO: Parallel each thread.
    cout << "iteration: " << iteration << endl;
    for (typename vector<unique_ptr<FusionThread<LabelType> > >::iterator fusion_thread_it = fusion_threads.begin(); fusion_thread_it != fusion_threads.end(); fusion_thread_it++) {
      cout << "thread: " << fusion_thread_it - fusion_threads.begin() << '\t' << (*fusion_thread_it)->getStatus() << endl;
      vector<LabelType> current_solution = solution_pool[cv_utils::drawFromArray(solution_confidence_vec)];
      (*fusion_thread_it)->setSolutionPool(solution_pool);
      (*fusion_thread_it)->setCurrentSolution(current_solution);
      (*fusion_thread_it)->runFusion();
      solution_pool[fusion_thread_it - fusion_threads.begin()] = (*fusion_thread_it)->getFusedSolution();

      //TODO: Better grabbing strategy
      energy_vec[fusion_thread_it - fusion_threads.begin()] = (*fusion_thread_it)->getFusedSolutionEnergy();
      vector<double>::const_iterator min_it = min_element(energy_vec.begin(), energy_vec.end());
      solution_confidence_vec.assign(pipeline_params.NUM_THREADS, 0);
      solution_confidence_vec[min_it - energy_vec.begin()] = 1;
    }
  }

  return solution_pool[min_element(energy_vec.begin(), energy_vec.end()) - energy_vec.begin()];
}
    
