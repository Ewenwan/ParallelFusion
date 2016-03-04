#include "SlaveFusionThread.h"

using namespace std;

void SlaveFusionThread::runFusion()
{
  proposal_generator_->setCurrentSolution(current_solution_);
  LabelSpace label_space = proposal_generator_->getProposal();
  fused_labels_ = fusion_solver_->solve(label_space);
}
