#ifndef PROPOSAL_GENERATOR_H__
#define PROPOSAL_GENERATOR_H__

#include <vector>

#include "LabelSpace.h"

template<class LabelType> class ProposalGenerator
{
 public:
  virtual void setCurrentSolution(const std::vector<LabelType> &current_solution) { current_solution_ = current_solution; };
  virtual LabelSpace<LabelType> getProposal() const = 0;
  virtual std::vector<LabelType> getInitialSolution() const = 0;
  virtual void writeSolution(const std::vector<LabelType> &solution, const int thread_index, const int iteration) const {}; //write temporary solution for visualization purpose
  
 protected:
  std::vector<LabelType> current_solution_;
};

#endif
