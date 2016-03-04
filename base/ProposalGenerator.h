#ifndef PROPOSAL_GENERATOR_H__
#define PROPOSAL_GENERATOR_H__

#include <vector>

#include "LabelSpace.h"

class ProposalGenerator
{
 public:
  virtual void setCurrentSolution(const std::vector<int> &current_solution) { current_solution_ = current_solution; };
  virtual LabelSpace getProposal() const = 0;
  
 protected:
  std::vector<int> current_solution_;
};

#endif
