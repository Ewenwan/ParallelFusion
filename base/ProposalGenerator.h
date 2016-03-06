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
  
 protected:
  std::vector<LabelType> current_solution_;
};

#endif
