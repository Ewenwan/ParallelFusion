#ifndef BASE_DATA_STRUCTURE_H__
#define BASE_DATA_STRUCTURE_H__

struct PipelineParams
{
public:
PipelineParams(const int _num_slave_threads, const int _num_fusion_iterations) : NUM_SLAVE_THREADS(_num_slave_threads), NUM_FUSION_ITERATIONS(_num_fusion_iterations) {};
  
  const int NUM_SLAVE_THREADS;
  const int NUM_FUSION_ITERATIONS;
}; 

#endif
