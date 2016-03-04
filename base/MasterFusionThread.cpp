#include "MasterFusionThread.h"

void MasterFusionThread::addSolution(const std::vector<int> &solution)
{
  solution_collection_ += LabelSpace(solution);
}

void MasterFusionThread::runFusion()
{
  current_solution_ = fusion_solver_->solve(solution_collection_);
}
