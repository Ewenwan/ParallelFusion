#ifndef PARALLEL_FUSION_PIPELINE_H__
#define PARALLEL_FUSION_PIPELINE_H__

#include <memory>
#include <vector>
#include <limits>
#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "BaseDataStructures.h"
#include "ProposalGenerator.h"
#include "FusionSolver.h"
#include "FusionThread.h"
#include "cv_utils/cv_utils.h"


template<class LabelType>
std::vector<LabelType> parallelFuse(std::vector<std::unique_ptr<FusionThread<LabelType> > > &fusion_threads, const PipelineParams &pipeline_params, const std::vector<LabelType> &initial_solution)
{
  typedef std::vector<LabelType> SolutionType;
  typedef std::vector<SolutionType> SolutionSet;

  SolutionType solution_pool(pipeline_params.NUM_THREADS, initial_solution);

  std::vector<double> energy_vec(pipeline_params.NUM_THREADS, std::numeric_limits<double>::max()); //Keep only the solution with lowest energy at the moment.

  std::vector<double> solution_confidence_vec(pipeline_params.NUM_THREADS, 1);
  for (int iteration = 0; iteration < pipeline_params.NUM_FUSION_ITERATIONS; iteration++) {
    //TODO: Parallel each thread.
    std::cout << "iteration: " << iteration << std::endl;
    int thread_index = 0;

    for (auto& fusion_thread_it: fusion_threads) {
      std::cout << "thread: " << thread_index << '\t' << fusion_thread_it->getStatus() << std::endl;
      SolutionType current_solution = solution_pool[cv_utils::drawFromArray(solution_confidence_vec)];
      fusion_thread_it->setSolutionPool(solution_pool);
      fusion_thread_it->setCurrentSolution(current_solution);
      fusion_thread_it->runFusion(thread_index, iteration);
      SolutionType fused_solution = fusion_thread_it->getFusedSolution();
      solution_pool[thread_index] = fused_solution;

      double energy = fusion_thread_it->getFusedSolutionEnergy();
      std::cout << "energy: " << energy << std::endl;
      //TODO: Better grabbing strategy

      energy_vec[thread_index] = energy;
      std::vector<double>::const_iterator min_it = min_element(energy_vec.begin(), energy_vec.end());
      solution_confidence_vec.assign(pipeline_params.NUM_THREADS, 0);
      solution_confidence_vec[min_it - energy_vec.begin()] = 1;
      thread_index++;
    }
  }

  return solution_pool[min_element(energy_vec.begin(), energy_vec.end()) - energy_vec.begin()];
}

#endif
