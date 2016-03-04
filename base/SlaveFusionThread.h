#ifndef SLAVE_FUSION_THREAD_H__
#define SLAVE_FUSION_THREAD_H__

#include <memory>

#include "ProposalGenerator.h"
#include "FusionSolver.h"

class SlaveFusionThread : FusionThread
{
 public:
 SlaveFusionThread(const shared_ptr<ProposalGenerator> &proposal_generator, const shared_ptr<FusionSolver> &fusion_solver) : proposal_generator_(proposal_generator), fusion_solver_(fusion_solver) {};

  void runFusion();
  std::vector<int> getFusedSolution() { return fused_solution_; };
  
 private:
  const shared_ptr<ProposalGenerator> &proposal_generator_;
  const shared_ptr<FusionSolver> &fusion_solver_;
  
  std::vector<int> fused_solution_;
};

#endif
