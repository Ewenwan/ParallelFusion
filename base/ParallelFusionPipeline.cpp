#include "ParallelFusionPipeline.h"

#include <vector>

using namespace std;

void parallelFuse(const shared_ptr<ProposalGenerator> &proposal_generator, const shared_ptr<FusionSolver> &fusion_solver, const PipelineParams &pipeline_params)
{
  vector<unique_ptr<SlaveFusionThread> > slave_fusion_threads;
  for (int slave_fusion_thread_index = 0; slave_fusion_thread_index < pipeline_params.NUM_SLAVE_THREADS; slave_fusion_thread_index++)
    slave_fusion_threads.push_back(move(SlaveFusionThread(proposal_generator, fusion_solver)));
  
  unique_ptr<MasterFusionThread> master_fusion_thread(fusion_solver);
  master_fusion_thread.setCurrentSolution(proposal_generator->getInitialSolution());
  for (int iteration = 0; iteration < pipeline_params.NUM_FUSION_ITERATIONS; iteration++) {
    //TODO: Parallel each thread.

    vector<int> current_solution = MasterFusionThread->getCurrentSolution();
    for (vector<unique_ptr<SlaveFusionThread> >::iterator slave_fusion_thread_it = slave_fusion_threads.begin(); slave_fusion_thread_it != slave_fusion_threads.end(); slave_fusion_thread_it++) {
      slave_fusion_thread_it->setCurrentSolution(current_solution);
      slave_fusion_thread_it->runFusion();
      master_fusion_thread->addSolution(slave_fusion_thread_it->getFusedSolution());
    }
    
    master_fusion_thread->runFusion();
  }
}
    
