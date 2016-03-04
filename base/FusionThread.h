#ifndef FUSION_THREAD_H__
#define FUSION_THREAD_H__

#include <vector>

#include "ProposalGenerator.h"
#include "FusionSolver.h"

class FusionThread
{
 public:
 FusionThread(const shared_ptr<ProposalGenerator> &proposal_generator, const shared_ptr<FusionSolver> &fusion_solver) : proposal_generator_(proposal_generator), fusion_solver_(fusion_solver) {};

  void runFusion();
  void setCurrentSolution() { current_solution_ = current_solution; };
  std::vector<int> getSolution() { return fused_solution_; };
  
 private:
  const shared_ptr<ProposalGenerator> &proposal_generator_;
  const shared_ptr<FusionSolver> &fusion_solver_;
  
  const std::vector<int> current_solution_;
  std::vector<int> fused_solution_;
}

#endif
