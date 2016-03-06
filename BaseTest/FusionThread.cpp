#include "FusionThread.h"


using namespace std;

void FusionThread::runFusion()
{
  if (is_master_) {
    if (fused_solution_.size() > 0)
      solution_collection_ += LabelSpace(fused_solution_);
    vector<vector<int> > solution_collection_vec = solution_collection_.getLabelSpace();
    //cout << solution_collection_vec.size() << endl;
    // for (int index = 0; index < solution_collection_vec.size(); index++)
    //   cout << solution_collection_vec[index].size() << endl;
    // exit(1);
    fused_solution_ = fusion_solver_->solve(solution_collection_, fused_solution_energy_);
  } else {
    //cout << "why" << endl;
    //solution_collection_ += LabelSpace(fused_solution_); //Maybe necessary
    proposal_generator_->setCurrentSolution(current_solution_);
    LabelSpace label_space = proposal_generator_->getProposal();
    fused_solution_ = fusion_solver_->solve(label_space, fused_solution_energy_);
  }

  updateStatus();
}

void FusionThread::setSolutionPool(const vector<vector<int> > &solution_pool)
{
  if (is_master_ && solution_pool.size() > 0) {
    solution_collection_.assign(solution_pool.begin()->size());
    for (vector<vector<int> >::const_iterator solution_it = solution_pool.begin(); solution_it != solution_pool.end(); solution_it++)
      solution_collection_ += LabelSpace(*solution_it);
  }
}
