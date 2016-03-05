#ifndef BASE_DATA_STRUCTURE_H__
#define BASE_DATA_STRUCTURE_H__

struct PipelineParams
{
public:
PipelineParams(const int _num_threads, const int _num_fusion_iterations) : NUM_THREADS(_num_threads), NUM_FUSION_ITERATIONS(_num_fusion_iterations) {};
  
  const int NUM_THREADS;
  const int NUM_FUSION_ITERATIONS;
}; 

#endif
