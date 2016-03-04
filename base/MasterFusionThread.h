#ifndef MASTER_FUSION_THREAD_H__
#define MASTER_FUSION_THREAD_H__

#include <memory>

#include "FusionThread.h"
#include "LabelSpace.h"

class MasterFusionThread : FusionThread
{
 public:
 MasterFusionThread(const shared_ptr<FusionSolver> &fusion_solver) : fusion_solver_(fusion_solver) {};

  void runFusion();
  void addSolution(const std::vector<int> &solution);
  std::vector<int> getCurrentSolution() { return current_solution_; };
  
 private:
  LabelSpace solution_collection_;
  const shared_ptr<FusionSolver> &fusion_solver_;
};

#endif
