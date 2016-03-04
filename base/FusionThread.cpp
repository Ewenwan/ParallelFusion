#include "FusionThread.h"

using namespace std;

void FusionThread::runFusion()
{
  proposal_generator_.setCurrentSolution(current_solution_);
  LabelSpace label_space = proposal_generator_.getProposal();
  fused_labels_ = fusion_solver_.solve(label_space);
}
