#ifndef FUSION_THREAD_H__
#define FUSION_THREAD_H__

#include <vector>
#include <memory>

#include "ProposalGenerator.h"
#include "FusionSolver.h"
#include "LabelSpace.h"

#include "../base/cv_utils/cv_utils.h"

template<class LabelType> class FusionThread
{
 public:
 FusionThread(const std::shared_ptr<ProposalGenerator<LabelType> > &proposal_generator, const std::shared_ptr<FusionSolver<LabelType> > &fusion_solver, const double MASTER_LIKELIHOOD) : proposal_generator_(proposal_generator), fusion_solver_(fusion_solver), MASTER_LIKELIHOOD_(MASTER_LIKELIHOOD) { updateStatus(); };
  
  void runFusion();  
  void setCurrentSolution(const std::vector<LabelType> &current_solution) { current_solution_ = current_solution; };

  std::vector<LabelType> getFusedSolution() { return fused_solution_; };
  double getFusedSolutionEnergy() { return fused_solution_energy_; };
  void setSolutionCollection(const LabelSpace<LabelType> &solution_collection) { if (is_master_) solution_collection_ = solution_collection; }; //Solution_collection is a label space.
  void setSolutionPool(const std::vector<std::vector<LabelType> > &solution_pool); //Solution_pool is a set of solutions.
  int getStatus() const { return is_master_; };
  
 private:
  std::vector<LabelType> current_solution_;
  std::vector<LabelType> fused_solution_;
  double fused_solution_energy_;

  const double MASTER_LIKELIHOOD_;
  bool is_master_;
  
  LabelSpace<LabelType> solution_collection_;
  const std::shared_ptr<ProposalGenerator<LabelType> > &proposal_generator_;
  const std::shared_ptr<FusionSolver<LabelType> > &fusion_solver_;

  
  void updateStatus() { is_master_ = cv_utils::randomProbability() < MASTER_LIKELIHOOD_; };
};

#endif
