#ifndef BASE_DATA_STRUCTURE_H__
#define BASE_DATA_STRUCTURE_H__

struct PipelineParams
{
public:
PipelineParams(const int _num_threads, const int _num_fusion_iterations, const int _num_proposal_solution_threshold) : NUM_THREADS(_num_threads), NUM_FUSION_ITERATIONS(_num_fusion_iterations), NUM_PROPOSAL_SOLUTIONS_THRESHOLD(_num_proposal_solution_threshold) {};
  
  const int NUM_THREADS;
  const int NUM_FUSION_ITERATIONS;
  const int NUM_PROPOSAL_SOLUTIONS_THRESHOLD;
}; 

#endif
